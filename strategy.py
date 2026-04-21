#!/usr/bin/env python3
"""
6h_WilliamsFractal_Donchian_Breakout_V1
Hypothesis: 6h strategy using 1-week Williams Fractals to identify major swing points and 1-day Donchian channels for breakout confirmation. Long when price breaks above 1d Donchian(20) high AND a weekly bullish fractal is present (confirmed with 2-bar delay). Short when price breaks below 1d Donchian(20) low AND a weekly bearish fractal is present. Uses volume confirmation (>1.3x 20-period 6h volume MA) and ATR-based stop (2.0*ATR). Designed for low trade frequency (target: 12-37 trades/year) to work in both bull and bear markets by capturing significant swing-driven breakouts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1w for fractals, 1d for Donchian)
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    if len(df_1w) < 10 or len(df_1d) < 30:
        return np.zeros(n)
    
    # === 1w Williams Fractals (requires 2-bar confirmation delay) ===
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    bearish_fractal, bullish_fractal = compute_williams_fractals(high_1w, low_1w)
    # Align with 2-bar additional delay for fractal confirmation
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1w, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1w, bullish_fractal, additional_delay_bars=2)
    
    # === 1d Donchian Channel (20-period) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # === 6h Indicators (primary timeframe) ===
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 30:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    volume_6h = df_6h['volume'].values
    
    # ATR (14-period) for stoploss
    tr1 = pd.Series(high_6h - low_6h)
    tr2 = pd.Series(np.abs(high_6h - np.roll(close_6h, 1)))
    tr3 = pd.Series(np.abs(low_6h - np.roll(close_6h, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # Volume MA (20-period) for confirmation
    vol_ma = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i])
            or np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i])
            or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_6h[i]
        vol = volume_6h[i]
        vol_ok = vol > 1.3 * vol_ma[i]  # volume confirmation
        
        if position == 0:
            # Long: 6h price breaks above 1d Donchian high + weekly bullish fractal + volume
            if price > donchian_high_aligned[i] and bullish_fractal_aligned[i] and vol_ok:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: 6h price breaks below 1d Donchian low + weekly bearish fractal + volume
            elif price < donchian_low_aligned[i] and bearish_fractal_aligned[i] and vol_ok:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Check stoploss
            if price < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit: price breaks below Donchian low or loss of volume/fractal
            elif price < donchian_low_aligned[i] or not vol_ok or not bullish_fractal_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Check stoploss
            if price > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit: price breaks above Donchian high or loss of volume/fractal
            elif price > donchian_high_aligned[i] or not vol_ok or not bearish_fractal_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsFractal_Donchian_Breakout_V1"
timeframe = "6h"
leverage = 1.0