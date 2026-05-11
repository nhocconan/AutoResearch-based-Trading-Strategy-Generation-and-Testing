#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike
Hypothesis: Combines Camarilla pivot levels (R1/S1) from 1d timeframe with 1d trend filter (EMA34) and volume confirmation to breakout on 4h chart. 
Long when price breaks above R1 in uptrend (close > EMA34) with volume spike, short when breaks below S1 in downtrend (close < EMA34) with volume spike. 
Uses 1d timeframe for structure and trend, 4h for execution. Designed for low trade frequency (20-50/year) to minimize fee drag in both bull and bear markets.
"""

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given period"""
    # Typical price
    typical = (high + low + close) / 3
    # Pivot point
    pivot = typical
    # Range
    range_val = high - low
    
    # Camarilla levels
    r1 = close + (range_val * 1.1 / 12)
    s1 = close - (range_val * 1.1 / 12)
    r2 = close + (range_val * 1.1 / 6)
    s2 = close - (range_val * 1.1 / 6)
    r3 = close + (range_val * 1.1 / 4)
    s3 = close - (range_val * 1.1 / 4)
    r4 = close + (range_val * 1.1 / 2)
    s4 = close - (range_val * 1.1 / 2)
    
    return r1, s1, r2, s2, r3, s3, r4, s4, pivot

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # 4h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 1d Camarilla Pivot Levels (from previous day) ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's data
    r1_1d, s1_1d, _, _, _, _, _, _, _ = calculate_camarilla(
        df_1d['high'].values, df_1d['low'].values, df_1d['close'].values
    )
    
    # Align 1d Camarilla levels to 4h timeframe (using previous day's levels)
    r1_1d_4h = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_4h = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # --- 1d Trend Filter (EMA34) ---
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_4h = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # --- Volume Spike Detection (20-period average on 4h) ---
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 40  # enough for EMA34 and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_1d_4h[i]) or np.isnan(s1_1d_4h[i]) or 
            np.isnan(ema_34_1d_4h[i]) or np.isnan(vol_ratio[i])):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation threshold
        volume_spike = vol_ratio[i] > 2.0
        
        # Trend direction: above EMA34 = uptrend, below EMA34 = downtrend
        uptrend = close[i] > ema_34_1d_4h[i]
        downtrend = close[i] < ema_34_1d_4h[i]
        
        if position == 0:
            # Long: price breaks above R1 in uptrend with volume spike
            if (close[i] > r1_1d_4h[i] and uptrend and volume_spike):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 in downtrend with volume spike
            elif (close[i] < s1_1d_4h[i] and downtrend and volume_spike):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: opposite break or trend reversal
            if position == 1:
                # Exit long: price breaks below S1 OR trend turns down
                if close[i] < s1_1d_4h[i] or (close[i] < ema_34_1d_4h[i] and ema_34_1d_4h[i] > ema_34_1d_4h[i-1]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price breaks above R1 OR trend turns up
                if close[i] > r1_1d_4h[i] or (close[i] > ema_34_1d_4h[i] and ema_34_1d_4h[i] < ema_34_1d_4h[i-1]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals