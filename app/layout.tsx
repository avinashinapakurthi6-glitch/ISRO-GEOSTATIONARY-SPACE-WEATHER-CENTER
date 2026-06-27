import type { Metadata } from "next";
import React from "react";
import "./globals.css";

export const metadata: Metadata = {
  title: "ISRO Geostationary Space Weather Center",
  description: "Energetic Particle Radiation Forecasting System — Satellite Safeguard Center for Geostationary Orbit",
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
