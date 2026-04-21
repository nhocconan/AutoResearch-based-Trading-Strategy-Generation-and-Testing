#!/usr/bin/env python3
"""
6h_WilliamsFractal_Breakout_12hTrend_VolumeConfirm_v1
Hypothesis: Williams Fractal breakout on 6h with 12h EMA50 trend filter and volume confirmation. Designed for low trade frequency (~15-30/year) to minimize fee drag and work in both bull/bear markets by only trading with the 12h trend. Uses 6h primary timeframe with 12h HTF for trend and volume context.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 60:
        return np.zeros(n)
    
    # === 12h trend filter: 50-period EMA ===
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # === 12h volume average (20-period) for spike detection ===
    volume_12h = df_12h['volume'].values
    vol_ma_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_ma_12h[np.isnan(vol_ma_12h)] = 1.0  # avoid division by zero
    vol_ratio_12h = volume_12h / vol_ma_12h
    vol_ratio_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ratio_12h)
    
    # === Williams Fractals on 6h (requires 5 bars: 2 left, center, 2 right) ===
    high = prices['high'].values
    low = prices['low'].values
    bearish_fractal, bullish_fractal = compute_williams_fractals(high, low)
    # Fractals need 2 extra bars for confirmation (center bar + 2 right bars)
    bearish_fractal_aligned = align_htf_to_ltf(prices, prices, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, prices, bullish_fractal, additional_delay_bars=2)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(ema_50_12h_aligned[i]) or
            np.isnan(vol_ratio_12h_aligned[i]) or
            np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        trend_12h = ema_50_12h_aligned[i]
        vol_spike = vol_ratio_12h_aligned[i]
        is_bearish_fractal = bearish_fractal_aligned[i] == 1
        is_bullish_fractal = bullish_fractal_aligned[i] == 1
        
        if position == 0:
            # Long: bullish fractal breakout + volume spike > 1.5 + price above 12h EMA50
            if is_bullish_fractal and vol_spike > 1.5 and price_close > trend_12h:
                signals[i] = 0.25
                position = 1
            # Short: bearish fractal breakout + volume spike > 1.5 + price below 12h EMA50
            elif is_bearish_fractal and vol_spike > 1.5 and price_close < trend_12h:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit when price crosses 12h EMA50 (trend change)
            if position == 1 and price_close < trend_12h:
                signals[i] = 0.0
                position = 0
            elif position == -1 and price_close > trend_12h:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_WilliamsFractal_Breakout_12hTrend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0