#!/usr/bin/env python3
"""
4h_Camarilla_Pivot_R1_S1_Breakout_1dTrend_VolumeS
Hypothesis: Uses Camarilla pivot levels (R1/S1) from daily data with 1d EMA34 trend filter and volume spike confirmation.
Enters long when price breaks above R1 in uptrend with volume confirmation, short when breaks below S1 in downtrend.
Targets 20-50 trades/year on 4h timeframe with disciplined risk management.
Works in bull markets via breakout continuation and bear markets via mean reversion at extreme levels.
"""

name = "4h_Camarilla_Pivot_R1_S1_Breakout_1dTrend_VolumeS"
timeframe = "4h"
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
    
    # === DAILY DATA FOR CAMARILLA PIVOTS AND TREND ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot levels (R1, S1)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Typical price for pivot calculation
    typical_price = (high_1d + low_1d + close_1d) / 3
    daily_range = high_1d - low_1d
    
    # Camarilla levels
    pivot = typical_price
    r1 = pivot + (1.1 * daily_range / 12)
    s1 = pivot - (1.1 * daily_range / 12)
    
    # Align Camarilla levels to 4h timeframe (wait for daily close)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # === VOLUME SPIKE CONFIRMATION (4h) ===
    # Volume ratio: current volume / 20-period average volume
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.where(vol_ma > 0, volume / vol_ma, 0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R1 in uptrend with volume spike
            if (close[i] > r1_aligned[i] and 
                ema34_1d_aligned[i] < close[i] and 
                vol_ratio[i] > 1.5):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 in downtrend with volume spike
            elif (close[i] < s1_aligned[i] and 
                  ema34_1d_aligned[i] > close[i] and 
                  vol_ratio[i] > 1.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price falls below S1 or trend changes
            if (close[i] < s1_aligned[i] or 
                ema34_1d_aligned[i] > close[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: price rises above R1 or trend changes
            if (close[i] > r1_aligned[i] or 
                ema34_1d_aligned[i] < close[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals