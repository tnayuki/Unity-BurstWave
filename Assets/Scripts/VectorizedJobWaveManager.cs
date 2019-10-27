using UnityEngine;
using Unity.Jobs;
using Unity.Burst;
using Unity.Collections;
using Unity.Mathematics;

public class VectorizedJobWaveManager : MonoBehaviour
{
    const float MESH_WIDTH = 32f;
    const int MESH_RANGE = 1024;
    Vector3[] m_Vertices;
    Mesh m_Mesh;

    Unity.Mathematics.Random random;
    NativeArray<float> inputBuffer;
    NativeArray<float> waveBuffer0;
    NativeArray<float> waveBuffer1;
    int activeBuffer = 0;

    void InitializeBuffers()
    {
        inputBuffer = new NativeArray<float>(MESH_RANGE*MESH_RANGE, Allocator.Persistent);
        waveBuffer0 = new NativeArray<float>(MESH_RANGE*MESH_RANGE, Allocator.Persistent);
        waveBuffer1 = new NativeArray<float>(MESH_RANGE*MESH_RANGE, Allocator.Persistent);
    }

    void DestroyBuffers()
    {
        waveBuffer1.Dispose();
        waveBuffer0.Dispose();
        inputBuffer.Dispose();
    }

    void Start()
    {
        m_Vertices = new Vector3[MESH_RANGE * MESH_RANGE];
        var vertices = m_Vertices;
        for (var y = 0; y < MESH_RANGE; ++y) {
            for (var x = 0; x < MESH_RANGE; ++x) {
                vertices[y*MESH_RANGE + x] = new Vector3(MESH_WIDTH/MESH_RANGE*(x-MESH_RANGE/2),
                                                         0f,
                                                         MESH_WIDTH/MESH_RANGE*(y-MESH_RANGE/2));
            }
        }
        var triangles = new int[(MESH_RANGE-1)*(MESH_RANGE-1)*2*3];
        for (var y = 0; y < MESH_RANGE-1; ++y) {
            for (var x = 0; x < MESH_RANGE-1; ++x) {
                triangles[(y*(MESH_RANGE-1) + x)*6 + 0] = (y+0)*(MESH_RANGE) + (x+0);
                triangles[(y*(MESH_RANGE-1) + x)*6 + 1] = (y+1)*(MESH_RANGE) + (x+0);
                triangles[(y*(MESH_RANGE-1) + x)*6 + 2] = (y+0)*(MESH_RANGE) + (x+1);
                triangles[(y*(MESH_RANGE-1) + x)*6 + 3] = (y+1)*(MESH_RANGE) + (x+0);
                triangles[(y*(MESH_RANGE-1) + x)*6 + 4] = (y+1)*(MESH_RANGE) + (x+1);
                triangles[(y*(MESH_RANGE-1) + x)*6 + 5] = (y+0)*(MESH_RANGE) + (x+1);
            }
        }
        var mesh = new Mesh();
        mesh.name = "water";
        mesh.indexFormat = UnityEngine.Rendering.IndexFormat.UInt32;
        mesh.vertices = vertices;
        mesh.triangles = triangles;
        mesh.RecalculateNormals();
        mesh.bounds = new Bounds(Vector3.zero, Vector3.one * 999999999f);

        m_Mesh = mesh;
        var mf = GetComponent<MeshFilter>();
        mf.sharedMesh = m_Mesh;

        InitializeBuffers();

        random = new Unity.Mathematics.Random();
        random.InitState(12345);
    }

    void OnDestroy()
    {
        DestroyBuffers();
    }

    void ClearBuffer(NativeArray<float> buffer)
    {
        for (var i = 0; i < buffer.Length; ++i) {
            buffer[i] = 0f;
        }
    }

    [BurstCompile]
    struct Job : IJob
    {
        [ReadOnly] public NativeArray<float> inputBuf;
        [ReadOnly] public NativeArray<float> prevBuf;
        public NativeArray<float> curBuf;

        public void Execute()
        {
            NativeArray<float4> inputBufV = inputBuf.Reinterpret<float, float4>();
            NativeArray<float4> prevBufV = prevBuf.Reinterpret<float, float4>();
            NativeArray<float4> curBufV = curBuf.Reinterpret<float, float4>();

            for (var i = 0; i < inputBuf.Length / 4; ++i) {
                var iv = new int4(i * 4 + 0, i * 4 + 1, i * 4 + 2, i * 4 + 3);

                var prevprevVal = curBufV[i];
                var prevVal = prevBufV[i];

                int4 centerXIdx = iv % MESH_RANGE;
                int4 centerYIdx = iv / MESH_RANGE;

                var leftIdx = math.select(
                    iv - 1, iv + (MESH_RANGE - 1), 
                    new bool4(centerXIdx.x == 0, centerXIdx.y == 0, centerXIdx.z == 0, centerXIdx.w == 0)
                );

                var rightIdx = math.select(
                    iv + 1, iv - (MESH_RANGE - 1),
                    new bool4(centerXIdx.x == (MESH_RANGE-1), centerXIdx.y == (MESH_RANGE-1), centerXIdx.z == (MESH_RANGE-1), centerXIdx.w == (MESH_RANGE-1))
                );

                var upIdx = math.select(
                    iv - MESH_RANGE, iv + ((MESH_RANGE - 1) * MESH_RANGE),
                    new bool4(centerYIdx.x == 0, centerYIdx.y == 0, centerYIdx.z == 0, centerYIdx.w == 0)
                );

                var downIdx = math.select(
                    iv + MESH_RANGE, iv - ((MESH_RANGE-1) * MESH_RANGE),
                    new bool4(centerYIdx.x == (MESH_RANGE-1), centerYIdx.y == (MESH_RANGE-1), centerYIdx.z == (MESH_RANGE-1), centerYIdx.w == (MESH_RANGE-1))
                );

                var leftVal = new float4(prevBuf[leftIdx.x], prevBuf[leftIdx.y], prevBuf[leftIdx.z], prevBuf[leftIdx.w]);
                var rightVal = new float4(prevBuf[rightIdx.x], prevBuf[rightIdx.y], prevBuf[rightIdx.z], prevBuf[rightIdx.w]);
                var upVal = new float4(prevBuf[upIdx.x], prevBuf[upIdx.y], prevBuf[upIdx.z], prevBuf[upIdx.w]);
                var downVal = new float4(prevBuf[downIdx.x], prevBuf[downIdx.y], prevBuf[downIdx.z], prevBuf[downIdx.w]);

                var c = 0.05f;
                var r = (leftVal + rightVal + upVal + downVal - (prevVal * 4)) * c - prevprevVal + 2 * prevVal;

                curBufV[i] = r;
                curBufV[i] = curBufV[i] * 0.999f;

                if (!inputBufV[i].Equals(float4.zero)) {
                    for (var j = 0; j < 4; ++j) {
                        if (inputBuf[i * 4 + j] != 0f) {
                            curBuf[i * 4 + j] = inputBuf[i * 4 + j];
                        }
                    }
                }
            }
        }
    }

    void DoWave(NativeArray<float> inputBuf,
                NativeArray<float> prevBuf,
                NativeArray<float> curBuf)
    {
        var job = new Job {
            inputBuf = inputBuf,
            prevBuf = prevBuf,
            curBuf = curBuf,
        };

        job.Run();
    }

    void ApplyWave(Mesh mesh, Vector3[] vertices, NativeArray<float> curBuf)
    {
        for (var i = 0; i < curBuf.Length; i++) {
            vertices[i].y = curBuf[i];
        }

        mesh.vertices = vertices;
        mesh.RecalculateNormals();
    }

    void Update()
    {
        ClearBuffer(inputBuffer);

        if (random.NextInt(1000) < 100) {
            var impulse = random.NextFloat2();
            inputBuffer[(int)(impulse.y*MESH_RANGE)*MESH_RANGE + (int)(impulse.x*MESH_RANGE)] = 1f;
        }

        var prevBuf = activeBuffer == 0 ? waveBuffer0 : waveBuffer1;
        var curBuf = activeBuffer == 0 ? waveBuffer1 : waveBuffer0;

        UnityEngine.Profiling.Profiler.BeginSample("DoWave");
        DoWave(inputBuffer, prevBuf, curBuf);
        UnityEngine.Profiling.Profiler.EndSample();

        ApplyWave(m_Mesh, m_Vertices, curBuf);

        activeBuffer = 1 - activeBuffer;
    }
}
