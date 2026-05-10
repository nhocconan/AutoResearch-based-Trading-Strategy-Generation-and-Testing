#!/usr/bin/env python3
# 1D_Cam_Pivot_Volume_Spike_Choice
# Hypothesis: Buy/sell at Camarilla S1/R1 levels on daily chart with volume spike confirmation.
# Uses 1-week trend filter to align with higher timeframe momentum. Designed for low turnover
# and high edge in both bull and bear markets by trading mean reversion within the trend.
# Targets 15-25 trades per year on 1d timeframe with position size 0.25.

name = "1D_Cam_Pivot_Volume_Spike_Choice"
timeframe = "1d"
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
    
    # Get 1d data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    # Using formula: R4 = C + (H-L)*1.1/2, R3 = C + (H-L)*1.1/4, etc.
    # We only need S1 and R1: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Calculate Camarilla R1 and S1 for each day
    r1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    s1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    
    # Align to intraday (but since we're on 1d, this is just for consistency)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # 1-week EMA(20) for trend direction
    ema_20_1w = pd.Series(df_1w['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Volume spike: current volume > 1.5 * 20-day average volume
    # Calculate 20-day average volume on 1d data
    vol_ma_20 = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    volume_spike = volume > (vol_ma_20_aligned * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Wait for volume MA to be valid
    
    for i in range(start_idx, n):
        if np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(ema_20_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price above/below weekly EMA20
        price_above_ema = close[i] > ema_20_1w_aligned[i]
        price_below_ema = close[i] < ema_20_1w_aligned[i]
        
        # Only trade in direction of weekly trend
        if position == 0:
            # Long at S1 in uptrend with volume spike
            if (price_above_ema and 
                close[i] <= s1_aligned[i] * 1.001 and  # Allow small slippage
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short at R1 in downtrend with volume spike
            elif (price_below_ema and 
                  close[i] >= r1_aligned[i] * 0.999 and  # Allow small slippage
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price reaches R1 or trend changes
            if (close[i] >= r1_aligned[i] * 0.999 or 
                not price_above_ema):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reaches S1 or trend changes
            if (close[i] <= s1_aligned[i] * 1.001 or 
                not price_below_ema):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals