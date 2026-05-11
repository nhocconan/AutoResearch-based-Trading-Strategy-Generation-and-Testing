#!/usr/bin/env python3
"""
1d_WilliamsFractal_Breakout_1wTrend_VolumeSpike
Hypothesis: Williams Fractal breakouts with 1w trend filter and volume spike confirmation. Works in bull markets (buy bullish fractal breakouts in uptrend) and bear markets (sell bearish fractal breakdowns in downtrend). Weekly trend filter avoids counter-trend trades. Volume spike confirms institutional interest. Target: 10-25 trades per year on 1d timeframe.
"""

name = "1d_WilliamsFractal_Breakout_1wTrend_VolumeSpike"
timeframe = "1d"
leverage = 1.0

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
    
    # === 1W Data for Trend Filter ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Trend filter: EMA34 on 1w close (needs 2-bar confirmation after close)
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w, additional_delay_bars=2)
    
    # === Williams Fractals on 1d ===
    bearish_fractal, bullish_fractal = compute_williams_fractals(high, low)
    # Fractals need 2 extra 1d bars after center bar for confirmation
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1w, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1w, bullish_fractal, additional_delay_bars=2)
    
    # Volume spike: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(bearish_fractal_aligned[i]) or 
            np.isnan(bullish_fractal_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: bullish fractal breakout AND uptrend (close > EMA34) AND volume spike
            if bullish_fractal_aligned[i] and close[i] > ema_34_1w_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: bearish fractal breakdown AND downtrend (close < EMA34) AND volume spike
            elif bearish_fractal_aligned[i] and close[i] < ema_34_1w_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: close crosses below EMA34 OR bearish fractal appears
            if close[i] < ema_34_1w_aligned[i] or bearish_fractal_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: close crosses above EMA34 OR bullish fractal appears
            if close[i] > ema_34_1w_aligned[i] or bullish_fractal_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals