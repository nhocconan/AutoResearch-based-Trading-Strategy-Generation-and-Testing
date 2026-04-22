#!/usr/bin/env python3
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
    
    # Load 1d data for Camarilla pivot levels (primary timeframe) and 1w data (HTF) - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 2 or len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot levels (R1, S1)
    # Pivot = (High + Low + Close) / 3
    # R1 = Pivot + (High - Low) * 1.1 / 12
    # S1 = Pivot - (High - Low) * 1.1 / 12
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot = (high_1d + low_1d + close_1d) / 3.0
    r1 = pivot + (high_1d - low_1d) * 1.1 / 12.0
    s1 = pivot - (high_1d - low_1d) * 1.1 / 12.0
    
    # Calculate weekly EMA34 for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 1d volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1w)
    vol_avg_20_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if data not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_avg_20_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above R1 with volume AND above weekly EMA34 (uptrend)
            if (close[i] > r1_aligned[i] and 
                volume[i] > 1.5 * vol_avg_20_aligned[i] and 
                close[i] > ema_34_1w_aligned[i]):
                signals[i] = 0.30
                position = 1
            # Short: Price breaks below S1 with volume AND below weekly EMA34 (downtrend)
            elif (close[i] < s1_aligned[i] and 
                  volume[i] > 1.5 * vol_avg_20_aligned[i] and 
                  close[i] < ema_34_1w_aligned[i]):
                signals[i] = -0.30
                position = -1
        else:
            # Exit: Price crosses back to opposite Camarilla level
            if position == 1:
                if close[i] < s1_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.30
            else:  # position == -1
                if close[i] > r1_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.30
    
    return signals

name = "12H_Camarilla_R1_S1_Breakout_1wEMA34_Trend_Volume"
timeframe = "12h"
leverage = 1.0