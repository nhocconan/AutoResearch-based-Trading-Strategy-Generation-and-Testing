#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_Camarilla_R1S1_Breakout_VolumeTrend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # === Weekly EMA Trend Filter ===
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema200_1w = pd.Series(close_1w).ewm(span=200, min_periods=200, adjust=False).mean().values
    
    # === Daily Data (price and volume) ===
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === Daily Close-based EMA for Trend (used in exit) ===
    close_series = pd.Series(close)
    ema50 = close_series.ewm(span=50, min_periods=50, adjust=False).mean().values
    ema200 = close_series.ewm(span=200, min_periods=200, adjust=False).mean().values
    
    # === Volume Filter (20-day average) ===
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Get values
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        vol_ma20_val = vol_ma20[i]
        ema50_val = ema50[i]
        ema200_val = ema200[i]
        ema50_1w_val = ema50_1w[i]
        ema200_1w_val = ema200_1w[i]
        
        # Skip if any value is NaN
        if (np.isnan(vol_ma20_val) or np.isnan(ema50_val) or 
            np.isnan(ema200_val) or np.isnan(ema50_1w_val) or 
            np.isnan(ema200_1w_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate daily pivot from previous day's data
        if i == 0:
            prev_high = high[0]
            prev_low = low[0]
            prev_close = close[0]
        else:
            prev_high = high[i-1]
            prev_low = low[i-1]
            prev_close = close[i-1]
        
        pivot = (prev_high + prev_low + prev_close) / 3
        range_val = prev_high - prev_low
        r1 = pivot + (range_val * 1.1 / 12)
        s1 = pivot - (range_val * 1.1 / 12)
        
        if position == 0:
            # Long: Break above R1 with volume confirmation and weekly uptrend
            if close_val > r1 and volume[i] > 2.0 * vol_ma20_val and ema50_1w_val > ema200_1w_val:
                signals[i] = 0.25
                position = 1
            # Short: Break below S1 with volume confirmation and weekly downtrend
            elif close_val < s1 and volume[i] > 2.0 * vol_ma20_val and ema50_1w_val < ema200_1w_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price returns below pivot OR weekly trend breaks down
            if close_val < pivot or ema50_1w_val < ema200_1w_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price returns above pivot OR weekly trend breaks up
            if close_val > pivot or ema50_1w_val > ema200_1w_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals