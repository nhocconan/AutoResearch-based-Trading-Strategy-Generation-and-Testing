#!/usr/bin/env python3
"""
6h_WilliamsFractal_Breakout_12hTrend_VolumeConfirm_v1
Hypothesis: Williams Fractal breakouts on 6h with 12h EMA50 trend filter and volume confirmation. 
Williams Fractals identify significant swing highs/lows that act as strong support/resistance. 
Breakouts above/below these levels with volume and trend alignment capture sustained moves. 
The 12h EMA50 provides trend filter that adapts to both bull and bear markets. 
Targeting 60-120 total trades over 4 years (15-30/year) to balance signal quality and fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop for HTF trend filter
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50 = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    
    # Calculate ATR(14) on 6h for volume normalization
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate Williams Fractals on 12h data (need 2 extra bars for confirmation)
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_12h['high'].values,
        df_12h['low'].values
    )
    # Align with 2 extra delay bars for fractal confirmation
    bearish_fractal_aligned = align_htf_to_ltf(
        prices, df_12h, bearish_fractal, additional_delay_bars=2
    )
    bullish_fractal_aligned = align_htf_to_ltf(
        prices, df_12h, bullish_fractal, additional_delay_bars=2
    )
    
    # Volume spike detection: volume > 2.0 * 20-period average volume
    avg_volume = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (2.0 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need sufficient data for all indicators)
    start_idx = max(100, 50, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_aligned[i]) or 
            np.isnan(bearish_fractal_aligned[i]) or
            np.isnan(bullish_fractal_aligned[i]) or
            np.isnan(atr[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # 12h trend filter (EMA50)
        uptrend = close[i] > ema_50_aligned[i]
        downtrend = close[i] < ema_50_aligned[i]
        
        # Long logic: price breaks above bullish fractal with volume spike + in uptrend
        if close[i] > bullish_fractal_aligned[i] and volume_spike[i] and uptrend:
            if position != 1:
                signals[i] = 0.25
                position = 1
            else:
                signals[i] = 0.25
        # Short logic: price breaks below bearish fractal with volume spike + in downtrend
        elif close[i] < bearish_fractal_aligned[i] and volume_spike[i] and downtrend:
            if position != -1:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = -0.25
        # Exit conditions: price returns to opposite fractal level or trend weakens
        elif position == 1 and (close[i] < bearish_fractal_aligned[i] or not uptrend):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (close[i] > bullish_fractal_aligned[i] or not downtrend):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsFractal_Breakout_12hTrend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0