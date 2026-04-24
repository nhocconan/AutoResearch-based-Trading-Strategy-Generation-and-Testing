#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Fractal breakout with 1w EMA50 trend filter and volume confirmation.
- Uses 1w EMA50 for primary trend alignment (HTF) to reduce whipsaw in ranging markets.
- Williams Fractals (2-bar confirmation) identify swing highs/lows for breakout entries.
- Volume spike (>2.0x 20-period average) filters low-conviction breakouts.
- Designed for 12h timeframe to target 50-150 total trades over 4 years (12-37/year).
- Works in bull/bear via trend filter: only long when above 1w EMA50, short when below.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data ONCE before loop for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1w EMA50 trend filter (needs 2 extra bars for confirmation as it's a lagging indicator)
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w, additional_delay_bars=2)
    
    # Get 1d data ONCE before loop for Williams Fractals (swing points)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Williams Fractals need 2 extra 1d bars after center bar for confirmation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    bearish_fractal, bullish_fractal = compute_williams_fractals(high_1d, low_1d)
    # Bearish fractal = swing high (for short entry), Bullish fractal = swing low (for long entry)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    # Volume confirmation: > 2.0x 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50) + 2  # +2 for fractal confirmation delay
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or 
            np.isnan(bearish_fractal_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: bullish fractal breakout with volume spike and above 1w EMA50
            if close[i] > bullish_fractal_aligned[i] and volume_spike[i] and close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: bearish fractal breakout with volume spike and below 1w EMA50
            elif close[i] < bearish_fractal_aligned[i] and volume_spike[i] and close[i] < ema_50_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price closes below bullish fractal (swing low) OR below 1w EMA50
            if close[i] < bullish_fractal_aligned[i] or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above bearish fractal (swing high) OR above 1w EMA50
            if close[i] > bearish_fractal_aligned[i] or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Williams_Fractal_Breakout_1wEMA50_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0