# 12H_1D_Camarilla_R1_S1_Breakout_1dTrend_Volume
# Hypothesis: On 12h timeframe, enter long when price breaks above Camarilla R1 level from previous 1d candle with 1d uptrend and volume confirmation.
# Short when price breaks below Camarilla S1 level with 1d downtrend and volume confirmation.
# Uses 1d trend filter to avoid counter-trend trades and Camarilla levels from 1d for precise entries.
# Target: 12-37 trades/year per symbol (50-150 total over 4 years).
# This strategy focuses on high-probability breakouts with trend and volume confirmation to minimize false signals.
# The 12h timeframe reduces trade frequency compared to lower timeframes, helping to avoid fee drag.
# Works in both bull and bear markets by following the 1d trend direction.

name = "12H_1D_Camarilla_R1_S1_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla levels and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for 1d: R1, S1 based on previous day
    # Typical price = (high + low + close) / 3
    typical_price = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    # Camarilla R1 = close + (range * 1.1/12)
    # Camarilla S1 = close - (range * 1.1/12)
    camarilla_r1 = close_1d + (range_1d * 1.1 / 12)
    camarilla_s1 = close_1d - (range_1d * 1.1 / 12)
    
    # 1d trend: EMA(34) on close
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_up = close_1d > ema_34
    
    # Volume confirmation: current volume > 1.5x 20-period average
    volume_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_avg * 1.5)
    
    # Align 1d indicators to 12h
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    trend_up_aligned = align_htf_to_ltf(prices, df_1d, trend_up)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or np.isnan(trend_up_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above Camarilla R1 + 1d uptrend + volume confirmation
            if close[i] > camarilla_r1_aligned[i] and trend_up_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below Camarilla S1 + 1d downtrend + volume confirmation
            elif close[i] < camarilla_s1_aligned[i] and not trend_up_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below Camarilla S1 (reversal) or trend changes
            if close[i] < camarilla_s1_aligned[i] or not trend_up_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above Camarilla R1 (reversal) or trend changes
            if close[i] > camarilla_r1_aligned[i] or trend_up_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals