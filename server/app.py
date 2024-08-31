from flask import Flask, request, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import io

app = Flask(__name__)
CORS(app)

@app.route('/generate-plot', methods=['POST'])
def generate_plot():
    # Ambil jumlah cluster dari input pengguna
    n_clusters = int(request.json.get('n_clusters'))

    # Membaca data dari file CSV
    df = pd.read_csv('koordinat_kelurahan_kabbdg.csv')
    df = df.dropna()

    # Data total volume sampah per jenis
    total_volume_sampah = {
        'total_volume_sampah': 1735.99,
    }

    # Total jumlah penduduk
    total_penduduk = df['jumlah_penduduk'].sum()

    # Estimasi volume sampah per jenis untuk setiap kelurahan
    for jenis, total_volume in total_volume_sampah.items():
        df[f'Est_Volume_{jenis}'] = (df['jumlah_penduduk'] / total_penduduk) * total_volume

    # Memilih fitur untuk clustering
    features = ['Latitude', 'Longitude'] + [f'Est_Volume_{jenis}' for jenis in total_volume_sampah.keys()]
    X = df[features].values

    # Normalisasi fitur
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Menggunakan KMeans untuk clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init='auto')
    df['Cluster_Label'] = kmeans.fit_predict(X_scaled)

    # Visualisasi hasil clustering
    plt.figure(figsize=(12, 6))
    plt.scatter(df['Latitude'], df['Longitude'], c=df['Cluster_Label'], cmap='viridis', marker='o')
    plt.xlabel('Latitude')
    plt.ylabel('Longitude')
    plt.title('Clustering Kelurahan Berdasarkan Estimasi Volume Sampah')
    plt.colorbar(label='Cluster Label')

    # Menentukan lokasi centroid cluster dalam skala asli
    centroids = kmeans.cluster_centers_
    centroids_original = scaler.inverse_transform(centroids)

    # Plot centroid dengan warna sesuai cluster
    colors = plt.cm.viridis(np.linspace(0, 1, n_clusters))  # Ambil warna dari colormap 'viridis'
    for i in range(n_clusters):
        plt.scatter(centroids_original[i, 0], centroids_original[i, 1],
                    color=colors[i], marker='x', s=100, label=f'Centroid {i}')

    plt.legend()

    # Simpan gambar ke dalam buffer untuk dikirim ke client
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()

    return send_file(buf, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
