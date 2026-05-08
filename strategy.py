#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Donchian20_Breakout_VolumeTrend"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data once for trend filter and Donchian reference
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 25:
        return np.zeros(n)
    
    # Daily EMA200 for trend filter
    close_1d = df_1d['close'].values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    trend_1d = (close_1d > ema200_1d).astype(float)
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # Daily Donchian channels (20-day high/low)
    high_20d = pd.Series(df_1d['high'].values).rolling(window=20, min_periods=20).max().values
    low_20d = pd.Series(df_1d['low'].values).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 4h timeframe
    donch_high_4h = align_htf_to_ltf(prices, df_1d, high_20d)
    donch_low_4h = align_htf_to_ltf(prices, df_1d, low_20d)
    
    # Volume spike: current volume > 2.5 * 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma20 * 2.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 25  # warmup for Donchian and volume MA
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(donch_high_4h[i]) or np.isnan(donch_low_4h[i]) or 
            np.isnan(trend_1d_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: price breaks above 20-day high with volume spike and daily uptrend
            long_cond = (close[i] > donch_high_4h[i] and vol_spike[i] and trend_1d_aligned[i] > 0.5)
            
            # Short entry: price breaks below 20-day low with volume spike and daily downtrend
            short_cond = (close[i] < donch_low_4h[i] and vol_spike[i] and trend_1d_aligned[i] < 0.5)
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below 20-day low (reversal signal)
            if close[i] < donch_low_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above 20-day high (reversal signal)
            if close[i] > donch_high_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Donchian(20) breakout on daily timeframe with volume spike confirmation and daily EMA200 trend filter on 4h timeframe.
# Enters long when price breaks above 20-day daily high with volume spike and daily uptrend (close > EMA200).
# Enters short when price breaks below 20-day daily low with volume spike and daily downtrend (close < EMA200).
# Exits when price reverses back through the opposite Donchian level.
# Uses daily Donchian channels to capture longer-term breaks, reducing false signals on 4h.
# Volume spike requirement ensures momentum behind the breakout.
# Trend filter ensures we only trade in the direction of the daily trend, reducing whipsaw.
# Works in bull markets (breakouts in uptrend) and bear markets (breakdowns in downtrend).
# Discrete sizing (0.25) minimizes churn. Targets ~20-40 trades/year on 4h timeframe.