#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_RSISlope_FilteredBreakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Get daily data for weekly pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === Weekly RSI (14-period) ===
    close_1w = df_1w['close'].values
    delta = np.diff(close_1w)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / np.where(avg_loss != 0, avg_loss, np.nan)
    rsi_1w = 100 - (100 / (1 + rs))
    # Align RSI to 6h
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    # === Weekly RSI Slope (3-period change) ===
    rsi_slope = np.zeros_like(rsi_1w)
    rsi_slope[3:] = (rsi_1w[3:] - rsi_1w[:-3]) / 3
    rsi_slope_aligned = align_htf_to_ltf(prices, df_1w, rsi_slope)
    
    # === Daily Pivot Points (previous week) for key levels ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous week's values (approx: 5 trading days)
    lookback = 5
    prev_high = np.roll(high_1d, lookback)
    prev_low = np.roll(low_1d, lookback)
    prev_close = np.roll(close_1d, lookback)
    
    # Weekly pivot point from prior week's OHLC
    pivot = (prev_high + prev_low + prev_close) / 3
    range_val = prev_high - prev_low
    
    # R1 and S1 levels
    r1 = pivot + (range_val * 1.1 / 12)
    s1 = pivot - (range_val * 1.1 / 12)
    
    # Align pivot levels to 6h
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    
    # === 6h Price and Volume ===
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume ratio (20-period average)
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    # Price change rate (6-period ROC for momentum)
    roc6 = np.zeros_like(close)
    roc6[6:] = (close[6:] - close[:-6]) / close[:-6] * 100
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Get values
        close_val = close[i]
        vol_ratio_val = vol_ratio[i]
        roc_val = roc6[i]
        rsi_val = rsi_1w_aligned[i]
        rsi_slope_val = rsi_slope_aligned[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        pivot_val = pivot_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(vol_ratio_val) or np.isnan(roc_val) or np.isnan(rsi_val) or 
            np.isnan(rsi_slope_val) or np.isnan(r1_val) or np.isnan(s1_val) or np.isnan(pivot_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: RSI rising from oversold, break above R1 with volume
            if (rsi_val < 30 and 
                rsi_slope_val > 0.5 and 
                close_val > r1_val and 
                vol_ratio_val > 1.5):
                signals[i] = 0.25
                position = 1
            # Short: RSI falling from overbought, break below S1 with volume
            elif (rsi_val > 70 and 
                  rsi_slope_val < -0.5 and 
                  close_val < s1_val and 
                  vol_ratio_val > 1.5):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: RSI overbought or price returns below pivot
            if rsi_val > 70 or close_val < pivot_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI oversold or price returns above pivot
            if rsi_val < 30 or close_val > pivot_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals