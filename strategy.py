#!/usr/bin/env python3
# 4H_Camarilla_R1_S1_Breakout_1dTrend_VolumeS
# Hypothesis: Uses Camarilla pivot levels (R1/S1) from 1-day timeframe. Enters long when price breaks above R1 with volume > 1.5x 20-period average in uptrend (close > EMA50). Enters short when price breaks below S1 with volume > 1.5x 20-period average in downtrend (close < EMA50). Exits when price returns to the pivot point (PP). Uses 1-day EMA50 for trend filter to avoid whipsaws. Targets 20-50 trades per year on 4h timeframe with position size 0.25.

name = "4H_Camarilla_R1_S1_Breakout_1dTrend_VolumeS"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1-day Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot Point (PP) = (High + Low + Close) / 3
    pp_1d = (high_1d + low_1d + close_1d) / 3.0
    # R1 = Close + 1.1 * (High - Low)
    r1_1d = close_1d + 1.1 * (high_1d - low_1d)
    # S1 = Close - 1.1 * (High - Low)
    s1_1d = close_1d - 1.1 * (high_1d - low_1d)
    
    # Calculate 1-day EMA(50) for trend direction
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d indicators to 4h timeframe
    pp_1d_aligned = align_htf_to_ltf(prices, df_1d, pp_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: 20-period average volume
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Warmup
    
    for i in range(start_idx, n):
        if np.isnan(pp_1d_aligned[i]) or np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price above/below 1d EMA50
        price_above_ema = close[i] > ema_50_1d_aligned[i]
        price_below_ema = close[i] < ema_50_1d_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirm = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Long entry: price breaks above R1 with volume confirmation in uptrend
            if (close[i] > r1_1d_aligned[i] and 
                volume_confirm and 
                price_above_ema):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below S1 with volume confirmation in downtrend
            elif (close[i] < s1_1d_aligned[i] and 
                  volume_confirm and 
                  price_below_ema):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price returns to or below pivot point (PP)
            if close[i] <= pp_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price returns to or above pivot point (PP)
            if close[i] >= pp_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals