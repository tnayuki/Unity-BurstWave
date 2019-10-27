using UnityEngine;
using Unity.Jobs;
using Unity.Burst;
using Unity.Collections;

public class JobWaveManager : MonoBehaviour
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
            for (var i = 0; i < inputBuf.Length; ++i) {
                if (inputBuf[i] != 0f) {
                    curBuf[i] = inputBuf[i];
                } else {
                    var prevprevVal = curBuf[i];
                    var prevVal = prevBuf[i];

                    int centerXIdx = i%MESH_RANGE;
                    var leftIdx = centerXIdx == 0 ? i + (MESH_RANGE-1) : i - 1;
                    var rightIdx = centerXIdx == (MESH_RANGE-1) ? i - (MESH_RANGE-1) : i + 1;

                    int centerYIdx = i/MESH_RANGE;
                    var upIdx = centerYIdx == 0 ? i + (MESH_RANGE-1)*MESH_RANGE : i - MESH_RANGE;
                    var downIdx = centerYIdx == (MESH_RANGE-1) ? i - (MESH_RANGE-1)*MESH_RANGE : i + MESH_RANGE;

                    var leftVal = prevBuf[leftIdx];
                    var rightVal = prevBuf[rightIdx];
                    var upVal = prevBuf[upIdx];
                    var downVal = prevBuf[downIdx];

                    var c = 0.4f;
                    curBuf[i] = c * (leftVal+rightVal+upVal+downVal-4*prevVal) - prevprevVal + 2*prevVal;
                    curBuf[i] = curBuf[i] * 0.999f;
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
