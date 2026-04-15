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
    
    # 1w SMA(50) for trend
    df_1w = get_htf_data(prices, '1w')
    sma_50_1w = pd.Series(df_1w['close'].values).rolling(window=50, min_periods=50).mean().values
    sma_50_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_50_1w)
    
    # 1d Donchian(20)
    df_1d = get_htf_data(prices, '1d')
    donch_high_20 = pd.Series(df_1d['high'].values).rolling(window=20, min_periods=20).max().values
    donch_low_20 = pd.Series(df_1d['low'].values).rolling(window=20, min_periods=20).min().values
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high_20)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low_20)
    
    # 1d volume: require > 1.5x median of last 20 days
    vol_median = pd.Series(volume).rolling(window=20, min_periods=1).median()
    vol_threshold = 1.5 * vol_median
    
    # 1d ATR(14) for volatility filter
    tr_1d = np.maximum(df_1d['high'].values - df_1d['low'].values,
                       np.maximum(np.abs(df_1d['high'].values - np.concatenate([[df_1d['close'][0]], df_1d['close'][:-1]])),
                                  np.abs(df_1d['low'].values - np.concatenate([[df_1d['close'][0]], df_1d['close'][:-1]]))))
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    atr_median = pd.Series(atr_1d_aligned).rolling(window=50, min_periods=1).median()
    vol_filter = atr_1d_aligned > 0.5 * atr_median
    
    signals = np.zeros(n)
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(sma_50_1w_aligned[i]) or np.isnan(donch_high_aligned[i]) or 
            np.isnan(donch_low_aligned[i]) or np.isnan(vol_threshold[i]) or 
            np.isnan(vol_filter[i])):
            continue
        
        # Long: close > 1w SMA(50) + breaks above Donchian high + volume + volatility filter
        if close[i] > sma_50_1w_aligned[i] and close[i] > donch_high_aligned[i] and \
           volume[i] > vol_threshold[i] and vol_filter[i]:
            signals[i] = 0.30
        
        # Short: close < 1w SMA(50) + breaks below Donchian low + volume + volatility filter
        elif close[i] < sma_50_1w_aligned[i] and close[i] < donch_low_aligned[i] and \
             volume[i] > vol_threshold[i] and vol_filter[i]:
            signals[i] = -0.30
        
        # Exit: close crosses back to 1w SMA(50) (mean reversion to trend)
        elif (i > 0 and 
              ((signals[i-1] == 0.30 and close[i] < sma_50_1w_aligned[i]) or
               (signals[i-1] == -0.30 and close[i] > sma_50_1w_aligned[i]))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "1d_Trend_Donchian_Volume_Filter"
timeframe = "1d"
leverage = 1.0