#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_LR_Channel_Breakout_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for linear regression channel and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Get 1d data for volume filter
    df_1d_vol = df_1d  # reuse same 1d data
    
    # Calculate linear regression channel on 1d close (20-period)
    close_1d = df_1d['close'].values
    lr_mid = np.full_like(close_1d, np.nan)
    lr_std = np.full_like(close_1d, np.nan)
    
    for i in range(20, len(close_1d)):
        y = close_1d[i-20:i]
        x = np.arange(20)
        if np.any(np.isnan(y)):
            continue
        slope, intercept = np.polyfit(x, y, 1)
        lr_mid[i] = slope * 19 + intercept  # predict last point
        residuals = y - (slope * x + intercept)
        lr_std[i] = np.std(residuals)
    
    # Upper and lower channel (2 std dev)
    lr_upper = lr_mid + 2 * lr_std
    lr_lower = lr_mid - 2 * lr_std
    
    # Trend filter: 1d EMA50
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume filter: current 1d volume > 1.5 * 20-period average
    vol_series = pd.Series(df_1d_vol['volume'].values)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_filter_1d = df_1d_vol['volume'].values > (vol_ma * 1.5)
    
    # Align all to 6h
    lr_upper_6h = align_htf_to_ltf(prices, df_1d, lr_upper)
    lr_lower_6h = align_htf_to_ltf(prices, df_1d, lr_lower)
    ema50_1d_6h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    volume_filter_1d_6h = align_htf_to_ltf(prices, df_1d_vol, volume_filter_1d)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(50, 20)  # Need enough data for EMA50 and LR
    
    for i in range(start_idx, n):
        if (np.isnan(lr_upper_6h[i]) or np.isnan(lr_lower_6h[i]) or
            np.isnan(ema50_1d_6h[i]) or np.isnan(volume_filter_1d_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        upper = lr_upper_6h[i]
        lower = lr_lower_6h[i]
        trend = ema50_1d_6h[i]
        vol_filter = volume_filter_1d_6h[i]
        
        if position == 0:
            # Enter long: break above upper channel with volume and above trend
            if close[i] > upper and close[i] > trend and vol_filter:
                signals[i] = 0.25
                position = 1
            # Enter short: break below lower channel with volume and below trend
            elif close[i] < lower and close[i] < trend and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: close below middle of channel (mean reversion)
            mid = (upper + lower) / 2
            if close[i] < mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: close above middle of channel (mean reversion)
            mid = (upper + lower) / 2
            if close[i] > mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals