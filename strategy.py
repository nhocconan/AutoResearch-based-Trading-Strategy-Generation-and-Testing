#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_Pivot_R1S1_Breakout_AdaptiveVWAP"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # === Daily Pivot Points (previous day) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's values for pivot calculation
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    
    # Set first values to avoid look-ahead
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    prev_close[0] = close_1d[0]
    
    # Classic pivot
    pivot = (prev_high + prev_low + prev_close) / 3
    range_val = prev_high - prev_low
    
    # R1 and S1 levels (standard)
    r1 = pivot + (range_val * 1.1 / 12)
    s1 = pivot - (range_val * 1.1 / 12)
    
    # Align to 6h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    
    # === Adaptive VWAP Deviation Filter ===
    # Calculate VWAP using typical price and volume
    typical_price = (prices['high'] + prices['low'] + prices['close']) / 3
    vwap_numerator = (typical_price * prices['volume']).cumsum()
    vwap_denominator = prices['volume'].cumsum()
    vwap = vwap_numerator / vwap_denominator
    vwap = vwap.values
    
    # VWAP deviation as percentage
    vwap_dev = (prices['close'].values - vwap) / vwap * 100
    
    # Adaptive threshold: 60-period std dev of VWAP deviation
    vwap_dev_series = pd.Series(vwap_dev)
    vwap_std = vwap_dev_series.rolling(window=60, min_periods=60).std().values
    vwap_threshold = 1.5 * vwap_std  # 1.5 standard deviations
    
    # === Momentum Filter: 6h ROC(12) ===
    close_series = pd.Series(prices['close'].values)
    roc12 = close_series.pct_change(periods=12) * 100  # ROC as percentage
    roc12 = roc12.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Get values
        close_val = prices['close'].iloc[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        pivot_val = pivot_aligned[i]
        vwap_dev_val = vwap_dev[i]
        vwap_thresh_val = vwap_threshold[i]
        roc12_val = roc12[i]
        
        # Skip if any value is NaN
        if (np.isnan(r1_val) or np.isnan(s1_val) or 
            np.isnan(pivot_val) or np.isnan(vwap_dev_val) or 
            np.isnan(vwap_thresh_val) or np.isnan(roc12_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above R1 with VWAP deviation above threshold and positive momentum
            if close_val > r1_val and vwap_dev_val > vwap_thresh_val and roc12_val > 0:
                signals[i] = 0.25
                position = 1
            # Short: Break below S1 with VWAP deviation below threshold and negative momentum
            elif close_val < s1_val and vwap_dev_val < -vwap_thresh_val and roc12_val < 0:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price returns below pivot OR VWAP deviation turns negative
            if close_val < pivot_val or vwap_dev_val < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price returns above pivot OR VWAP deviation turns positive
            if close_val > pivot_val or vwap_dev_val > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals