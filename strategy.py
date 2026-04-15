#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly ATR for volatility filter
    df_1w = get_htf_data(prices, '1w')
    tr_1w = np.maximum(df_1w['high'].values - df_1w['low'].values,
                       np.maximum(np.abs(df_1w['high'].values - np.concatenate([[df_1w['close'][0]], df_1w['close'][:-1]])),
                                  np.abs(df_1w['low'].values - np.concatenate([[df_1w['close'][0]], df_1w['close'][:-1]]))))
    atr_1w = pd.Series(tr_1w).rolling(window=14, min_periods=14).mean().values
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    
    # Daily ATR for volatility filter (used for exit)
    df_1d = get_htf_data(prices, '1d')
    tr_1d = np.maximum(df_1d['high'].values - df_1d['low'].values,
                       np.maximum(np.abs(df_1d['high'].values - np.concatenate([[df_1d['close'][0]], df_1d['close'][:-1]])),
                                  np.abs(df_1d['low'].values - np.concatenate([[df_1d['close'][0]], df_1d['close'][:-1]]))))
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # 12h Bollinger Bands (20, 2)
    sma_12h = pd.Series(close).rolling(window=20, min_periods=20).mean()
    std_12h = pd.Series(close).rolling(window=20, min_periods=20).std()
    upper_12h = sma_12h + 2 * std_12h
    lower_12h = sma_12h - 2 * std_12h
    
    # Volume confirmation: current > 2.0x median of last 20 bars (more stringent)
    vol_median = pd.Series(volume).rolling(window=20, min_periods=1).median()
    vol_threshold = 2.0 * vol_median
    
    # Volatility filter: require ATR > 0.5 * median ATR (more stringent)
    atr_median = pd.Series(atr_1w_aligned).rolling(window=50, min_periods=1).median()
    vol_filter = atr_1w_aligned > 0.5 * atr_median
    
    signals = np.zeros(n)
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_12h[i]) or np.isnan(lower_12h[i]) or 
            np.isnan(vol_threshold[i]) or np.isnan(vol_filter[i]) or
            np.isnan(atr_1d_aligned[i])):
            continue
        
        # Long: close breaks above upper band + volume + volatility filter
        if close[i] > upper_12h[i] and volume[i] > vol_threshold[i] and vol_filter[i]:
            signals[i] = 0.25
        
        # Short: close breaks below lower band + volume + volatility filter
        elif close[i] < lower_12h[i] and volume[i] > vol_threshold[i] and vol_filter[i]:
            signals[i] = -0.25
        
        # Exit: price crosses back inside bands with daily ATR filter to avoid whipsaw
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and close[i] < upper_12h[i]) or
               (signals[i-1] == -0.25 and close[i] > lower_12h[i])) and
              atr_1d_aligned[i] > 0.2 * pd.Series(atr_1d_aligned).rolling(window=20, min_periods=1).median()[i]):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "12h_Bollinger_Breakout_Volume_VolFilter"
timeframe = "12h"
leverage = 1.0