#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_Aroon_Dip_Buy_Trend_Follow"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === Daily Aroon Indicator (25-period) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    period = 25
    
    # Days since highest high
    def days_since_high(high_arr, p):
        n_arr = len(high_arr)
        since_high = np.full(n_arr, np.nan)
        for i in range(p, n_arr):
            window = high_arr[i-p+1:i+1]
            if len(window) == p:
                max_idx = np.argmax(window)
                since_high[i] = p - 1 - max_idx
        return since_high
    
    # Days since lowest low
    def days_since_low(low_arr, p):
        n_arr = len(low_arr)
        since_low = np.full(n_arr, np.nan)
        for i in range(p, n_arr):
            window = low_arr[i-p+1:i+1]
            if len(window) == p:
                min_idx = np.argmin(window)
                since_low[i] = p - 1 - min_idx
        return since_low
    
    days_since_high_val = days_since_high(high_1d, period)
    days_since_low_val = days_since_low(low_1d, period)
    
    # Aroon Up = ((period - days since high) / period) * 100
    aroon_up = np.where(~np.isnan(days_since_high_val), 
                        ((period - days_since_high_val) / period) * 100, np.nan)
    # Aroon Down = ((period - days since low) / period) * 100
    aroon_down = np.where(~np.isnan(days_since_low_val), 
                          ((period - days_since_low_val) / period) * 100, np.nan)
    
    # Aroon Oscillator = Aroon Up - Aroon Down
    aroon_osc = aroon_up - aroon_down
    
    # Align to 6h timeframe
    aroon_osc_aligned = align_htf_to_ltf(prices, df_1d, aroon_osc)
    
    # === 6h Price and Volume Filters ===
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 60-period EMA for trend filter
    close_series = pd.Series(close)
    ema60 = close_series.ewm(span=60, min_periods=60, adjust=False).mean().values
    
    # Volume ratio (current vs 20-period average)
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Get values
        close_val = close[i]
        ema60_val = ema60[i]
        aroon_val = aroon_osc_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(close_val) or np.isnan(ema60_val) or 
            np.isnan(aroon_val) or np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Aroon oscillator > 50 (strong uptrend) + price above EMA60 + volume confirmation
            if aroon_val > 50 and close_val > ema60_val and vol_ratio_val > 1.5:
                signals[i] = 0.25
                position = 1
            # Short: Aroon oscillator < -50 (strong downtrend) + price below EMA60 + volume confirmation
            elif aroon_val < -50 and close_val < ema60_val and vol_ratio_val > 1.5:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Aroon turns negative OR price breaks below EMA60
            if aroon_val < 0 or close_val < ema60_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Aroon turns positive OR price breaks above EMA60
            if aroon_val > 0 or close_val > ema60_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals