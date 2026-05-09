#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_Camarilla_R1S1_Breakout_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate weekly EMA(34) for trend filter
    close_1w = pd.Series(df_1w['close'].values)
    ema34_1w = close_1w.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Calculate weekly volume average (20-period)
    vol_1w = pd.Series(df_1w['volume'].values)
    vol_ma20_1w = vol_1w.rolling(window=20, min_periods=20).mean().values
    vol_ma20_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma20_1w)
    
    # Current volume for confirmation
    vol_series = pd.Series(volume)
    vol_ma20_current = vol_series.rolling(window=20, min_periods=20).mean().values
    
    # Daily Camarilla pivot levels
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # R1 = Pivot + (Range * 1.1 / 12)
    # S1 = Pivot - (Range * 1.1 / 12)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    close_series = pd.Series(close)
    pivot = (high_series + low_series + close_series) / 3
    range_hl = high_series - low_series
    camarilla_r1 = pivot + (range_hl * 1.1 / 12)
    camarilla_s1 = pivot - (range_hl * 1.1 / 12)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema34_1w_aligned[i]) or np.isnan(vol_ma20_1w_aligned[i]) or 
            np.isnan(vol_ma20_current[i]) or np.isnan(camarilla_r1[i]) or 
            np.isnan(camarilla_s1[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 1.5 * vol_ma20_current[i]
        
        if position == 0:
            # Long: Break above R1 with volume and above weekly EMA trend
            if close[i] > camarilla_r1[i] and vol_ok and close[i] > ema34_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Break below S1 with volume and below weekly EMA trend
            elif close[i] < camarilla_s1[i] and vol_ok and close[i] < ema34_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price falls below S1 or trend reversal
            if close[i] < camarilla_s1[i] or close[i] < ema34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price rises above R1 or trend reversal
            if close[i] > camarilla_r1[i] or close[i] > ema34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals