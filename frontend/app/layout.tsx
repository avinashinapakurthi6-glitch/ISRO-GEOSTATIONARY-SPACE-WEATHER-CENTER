import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "ISRO Geostationary Space Weather Center",
  description: "Energetic Particle Radiation Forecasting System - Satellite Safeguard Operations",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}

