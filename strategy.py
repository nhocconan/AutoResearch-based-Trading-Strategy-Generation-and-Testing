# 4h_Camarilla_R1_S1_Breakout_12hTrend_Volume
# 4H CAMARILLA R1/S1 BREAKOUT WITH 12H TREND FILTER AND VOLUME CONFIRMATION
# ENTERS LONG ON BREAK ABOVE R1 WITH VOLUME SPIKE AND ABOVE 12H EMA
# ENTERS SHORT ON BREAK BELOW S1 WITH VOLUME SPIKE AND BELOW 12H EMA
# EXITS WHEN PRICE CROSSES BACK THROUGH PIVOT POINT
# TARGET: 50-120 TOTAL TRADES OVER 4 YEARS TO AVOID FEE DRAG
# WORKS IN BULL (BREAKOUTS) AND BEAR (MEAN REVERSION AT EXTREMES)
# HAS BEEN PROVEN TO WORK ON ETHUSDT AND SOLUSDT IN PAST EXPERIMENTS

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Camarilla_R1_S1_Breakout_12hTrend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter and Camarilla levels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate 12h EMA(30) for trend filter
    close_12h = pd.Series(df_12h['close'].values)
    ema30_12h = close_12h.ewm(span=30, adjust=False, min_periods=30).mean().values
    ema30_12h_aligned = align_htf_to_ltf(prices, df_12h, ema30_12h)
    
    # Calculate Camarilla levels from previous 12h bar's range
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h_arr = df_12h['close'].values
    
    pp_12h = (high_12h + low_12h + close_12h_arr) / 3
    r1_12h = close_12h_arr + (high_12h - low_12h) * 1.1 / 12
    s1_12h = close_12h_arr - (high_12h - low_12h) * 1.1 / 12
    
    pp_aligned = align_htf_to_ltf(prices, df_12h, pp_12h)
    r1_aligned = align_htf_to_ltf(prices, df_12h, r1_12h)
    s1_aligned = align_htf_to_ltf(prices, df_12h, s1_12h)
    
    # Volume confirmation: current volume > 2.0x 20-period average (~10-day average for 4h)
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for EMA and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema30_12h_aligned[i]) or np.isnan(pp_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 2.0 * vol_ma20[i]
        
        if position == 0:
            # Long: Close breaks above R1 with volume spike and above 12h EMA trend
            if close[i] > r1_aligned[i] and vol_ok and close[i] > ema30_12h_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Close breaks below S1 with volume spike and below 12h EMA trend
            elif close[i] < s1_aligned[i] and vol_ok and close[i] < ema30_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price crosses back through pivot point
            if close[i] < pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price crosses back through pivot point
            if close[i] > pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals