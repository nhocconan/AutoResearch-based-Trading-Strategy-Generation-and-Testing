#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation
# Uses 1d primary timeframe with 1w HTF for trend alignment to reduce whipsaw.
# Donchian channel from 1w timeframe acts as strong support/resistance.
# Breakouts in direction of 1w trend with volume spike capture institutional moves.
# Designed for low trade frequency (7-25/year) to minimize fee drag in 1d timeframe.
# Works in both bull and bear markets by following the 1w trend direction only.

name = "1d_Donchian20_1wEMA50_Trend_Volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for Donchian channel and EMA50 trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    
    # Calculate Donchian channel (20) on 1w
    # Based on previous 20 completed 1w bars
    prev_high = df_1w['high'].shift(1).rolling(window=20, min_periods=20).max().values
    prev_low = df_1w['low'].shift(1).rolling(window=20, min_periods=20).min().values
    
    # Calculate EMA(50) on 1w for trend filter
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align Donchian levels and EMA to 1d timeframe (wait for completed 1w bar)
    upper_aligned = align_htf_to_ltf(prices, df_1w, prev_high)
    lower_aligned = align_htf_to_ltf(prices, df_1w, prev_low)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation (2.0x 20-period average) on 1d
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for EMA and Donchian calculations)
    start_idx = 70  # max(50 for EMA, 20 for Donchian +20 for volume MA +1 for shift)
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above upper Donchian + 1w uptrend + volume spike
            if close[i] > upper_aligned[i] and close[i] > ema_50_1w_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below lower Donchian + 1w downtrend + volume spike
            elif close[i] < lower_aligned[i] and close[i] < ema_50_1w_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price returns below EMA50 (trend reversal) or reaches lower Donchian (mean reversion)
            if close[i] < ema_50_1w_aligned[i] or close[i] < lower_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price returns above EMA50 (trend reversal) or reaches upper Donchian (mean reversion)
            if close[i] > ema_50_1w_aligned[i] or close[i] > upper_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals