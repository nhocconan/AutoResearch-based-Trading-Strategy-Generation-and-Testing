#!/usr/bin/env python3
name = "12H_Keltner_Channel_Trend_Pullback"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter and Keltner parameters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 20-day EMA for trend filter
    ema20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate daily ATR(10) for Keltner channels
    tr_1d = np.maximum(
        high_1d[1:] - low_1d[1:],
        np.maximum(
            np.abs(high_1d[1:] - close_1d[:-1]),
            np.abs(low_1d[1:] - close_1d[:-1])
        )
    )
    tr_1d = np.concatenate([[np.nan], tr_1d])  # align with original index
    atr10_1d = pd.Series(tr_1d).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Calculate Keltner channels: EMA(20) ± 2*ATR(10)
    upper_keltner = ema20_1d + 2 * atr10_1d
    lower_keltner = ema20_1d - 2 * atr10_1d
    
    # Align to 12h timeframe
    ema20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema20_1d)
    upper_keltner_aligned = align_htf_to_ltf(prices, df_1d, upper_keltner)
    lower_keltner_aligned = align_htf_to_ltf(prices, df_1d, lower_keltner)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data for calculations
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if EMA data not ready
        if np.isnan(ema20_1d_aligned[i]) or np.isnan(upper_keltner_aligned[i]) or np.isnan(lower_keltner_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.5x 10-period average volume
        if i >= 10:
            avg_volume = np.mean(volume[i-10:i])
            volume_confirm = volume[i] > avg_volume * 1.5
        else:
            volume_confirm = False
        
        if position == 0:
            # Enter long: price touches lower Keltner band + uptrend + volume confirmation
            if low[i] <= lower_keltner_aligned[i] and close[i] > ema20_1d_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Enter short: price touches upper Keltner band + downtrend + volume confirmation
            elif high[i] >= upper_keltner_aligned[i] and close[i] < ema20_1d_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses above EMA(20) or touches upper band
            if close[i] >= ema20_1d_aligned[i] or high[i] >= upper_keltner_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses below EMA(20) or touches lower band
            if close[i] <= ema20_1d_aligned[i] or low[i] <= lower_keltner_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals