#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Anchored_VWAP_Deviation_1dTrend_Filter"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_6h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate VWAP reset at each 1d boundary (start of day)
    vwap = np.zeros(n)
    vwap_sum = 0.0
    vol_sum = 0.0
    prev_date = None
    
    for i in range(n):
        current_date = pd.Timestamp(prices['open_time'].iloc[i]).date()
        if prev_date is None or current_date != prev_date:
            # Reset at start of new day
            vwap_sum = 0.0
            vol_sum = 0.0
            prev_date = current_date
        
        typical_price = (high[i] + low[i] + close[i]) / 3.0
        vwap_sum += typical_price * volume[i]
        vol_sum += volume[i]
        
        if vol_sum > 0:
            vwap[i] = vwap_sum / vol_sum
        else:
            vwap[i] = close[i]
    
    # VWAP deviation as percentage
    vwap_dev = (close - vwap) / vwap * 100.0
    
    # Calculate 20-period standard deviation of VWAP deviation for dynamic bands
    vwap_dev_ma = pd.Series(vwap_dev).rolling(window=20, min_periods=20).mean().values
    vwap_dev_std = pd.Series(vwap_dev).rolling(window=20, min_periods=20).std().values
    
    # Dynamic bands: 2 standard deviations
    upper_band = vwap_dev_ma + 2.0 * vwap_dev_std
    lower_band = vwap_dev_ma - 2.0 * vwap_dev_std
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema50_6h[i]) or np.isnan(upper_band[i]) or 
            np.isnan(lower_band[i]) or np.isnan(vwap_dev[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price deviates significantly below VWAP (oversold) in uptrend
            if vwap_dev[i] < lower_band[i] and close[i] > ema50_6h[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price deviates significantly above VWAP (overbought) in downtrend
            elif vwap_dev[i] > upper_band[i] and close[i] < ema50_6h[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price returns to VWAP or trend turns down
            if vwap_dev[i] > vwap_dev_ma[i] or close[i] < ema50_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price returns to VWAP or trend turns up
            if vwap_dev[i] < vwap_dev_ma[i] or close[i] > ema50_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals