#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_PearsonCorr_Momentum_1dTrend_Volume"
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
    
    # Get 1d data for trend filter and momentum calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 1d close for trend filter
    close_1d = df_1d['close'].values
    ema_10_1d = pd.Series(close_1d).ewm(span=10, adjust=False, min_periods=10).mean().values
    ema_10_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_10_1d)
    
    # 6h Pearson correlation momentum (5-period correlation with linear trend)
    # Calculate rolling correlation between close and time index
    x = np.arange(5)  # [0,1,2,3,4]
    x_mean = np.mean(x)
    x_var = np.sum((x - x_mean) ** 2)
    
    corr_values = np.full(n, np.nan)
    for i in range(4, n):
        y = close[i-4:i+1]
        y_mean = np.mean(y)
        # Pearson correlation formula: cov(x,y) / (std(x)*std(y))
        cov_xy = np.sum((x - x_mean) * (y - y_mean))
        y_var = np.sum((y - y_mean) ** 2)
        if y_var > 0 and x_var > 0:
            corr_values[i] = cov_xy / (np.sqrt(x_var * y_var))
        else:
            corr_values[i] = 0.0
    
    # 6h volume moving average for spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 10)  # Need 20 for volume MA, 10 for EMA
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(corr_values[i]) or np.isnan(ema_10_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        corr = corr_values[i]
        ema_1d = ema_10_1d_aligned[i]
        vol = volume[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # Enter long: Positive momentum (corr > 0.3) AND price > 1d EMA10 (uptrend) AND volume > 2x average
            if corr > 0.3 and close[i] > ema_1d and vol > 2.0 * vol_ma_val:
                signals[i] = 0.25
                position = 1
            # Enter short: Negative momentum (corr < -0.3) AND price < 1d EMA10 (downtrend) AND volume > 2x average
            elif corr < -0.3 and close[i] < ema_1d and vol > 2.0 * vol_ma_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Momentum turns negative (corr < 0) OR trend reverses (price < 1d EMA10)
            if corr < 0.0 or close[i] < ema_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Momentum turns positive (corr > 0) OR trend reverses (price > 1d EMA10)
            if corr > 0.0 or close[i] > ema_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals