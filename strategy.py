#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) + 1d EMA200 trend filter + volume confirmation
# Elder Ray measures bull/bear power relative to EMA13; EMA200 defines higher timeframe trend
# Works in bull/bear: EMA200 filter ensures we only take Elder Ray signals in direction of higher TF trend
# Volume confirmation reduces false signals. Target: 50-150 total trades over 4 years (12-37/year)

name = "6h_1d_elder_ray_ema200_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA200 for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema200_1d = close_1d.ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Calculate 6h EMA13 for Elder Ray
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Calculate 20-period average volume for confirmation
    volume_s = pd.Series(volume)
    avg_volume = volume_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(ema200_1d_aligned[i]) or np.isnan(ema13[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x 20-period average
        volume_confirmed = volume[i] > 1.3 * avg_volume[i]
        
        if position == 1:  # Long position
            # Exit: bear power > 0 (selling pressure) OR close below EMA13 (trend change)
            if bear_power[i] > 0 or close[i] < ema13[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: bull power < 0 (no buying pressure) OR close above EMA13 (trend change)
            if bull_power[i] < 0 or close[i] > ema13[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic with volume confirmation and Elder Ray + 1d EMA200 filter
            if volume_confirmed:
                # Long entry: bull power > 0 AND close > EMA13 AND price > 1d EMA200 (bullish in uptrend)
                if bull_power[i] > 0 and close[i] > ema13[i] and close[i] > ema200_1d_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short entry: bear power < 0 AND close < EMA13 AND price < 1d EMA200 (bearish in downtrend)
                elif bear_power[i] < 0 and close[i] < ema13[i] and close[i] < ema200_1d_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals