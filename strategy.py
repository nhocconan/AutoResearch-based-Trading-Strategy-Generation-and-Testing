#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R3/S3 breakout from 1d pivot levels + 1d EMA(34) trend filter + volume spike (>2x 20-period average)
- Uses tighter Camarilla R3/S3 levels (more significant than R1/S1) to reduce false breakouts and lower trade frequency
- 1d EMA(34) ensures trades align with daily trend to avoid counter-trend whipsaws
- Volume spike (>2x average) validates breakout with institutional participation
- Designed for 4h timeframe targeting 19-50 trades/year (75-200 over 4 years) to avoid fee drag
- Works in both bull and bear markets by trading with 1d trend from daily breakouts
"""

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
    
    # Get 1d data for Camarilla pivot levels and EMA trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA(34) for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla pivot levels for 1d timeframe
    # Pivot point = (H + L + C) / 3
    pivot = (high_1d + low_1d + close_1d) / 3.0
    # R3 = C + ((H-L) * 1.1 / 4)
    r3 = close_1d + ((high_1d - low_1d) * 1.1 / 4.0)
    # S3 = C - ((H-L) * 1.1 / 4)
    s3 = close_1d - ((high_1d - low_1d) * 1.1 / 4.0)
    
    # Align Camarilla levels to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    
    # Volume confirmation: > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 34)  # EMA and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Camarilla breakout conditions
        # Long: price breaks above R3 level
        # Short: price breaks below S3 level
        long_breakout = close[i] > r3_aligned[i]
        short_breakout = close[i] < s3_aligned[i]
        
        # Trend filter: price > EMA for long, price < EMA for short
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        
        if position == 0:
            # Long conditions: breakout above R3, uptrend, volume spike
            long_signal = (long_breakout and 
                          uptrend and
                          volume[i] > 2.0 * vol_ma[i])
            
            # Short conditions: breakout below S3, downtrend, volume spike
            short_signal = (short_breakout and 
                           downtrend and
                           volume[i] > 2.0 * vol_ma[i])
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: price returns to opposite Camarilla level or trend reversal
            exit_signal = False
            
            if position == 1:
                # Exit long: price falls below S3 level or trend turns down
                if (close[i] < s3_aligned[i] or 
                    not uptrend):  # Trend reversal
                    exit_signal = True
            elif position == -1:
                # Exit short: price rises above R3 level or trend turns up
                if (close[i] > r3_aligned[i] or 
                    not downtrend):  # Trend reversal
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0