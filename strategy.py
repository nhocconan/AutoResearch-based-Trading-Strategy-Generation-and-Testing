#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1D data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Daily 50 EMA for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Daily ATR for volatility filter
    tr_1d = np.maximum(df_1d['high'].values - df_1d['low'].values,
                       np.maximum(np.abs(df_1d['high'].values - np.concatenate([[df_1d['close'][0]], df_1d['close'][:-1]])),
                                  np.abs(df_1d['low'].values - np.concatenate([[df_1d['close'][0]], df_1d['close'][:-1]]))))
    atr_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # 4-hour Bollinger Bands (20, 2)
    sma_4h = pd.Series(close).rolling(window=20, min_periods=20).mean()
    std_4h = pd.Series(close).rolling(window=20, min_periods=20).std()
    upper = sma_4h + 2 * std_4h
    lower = sma_4h - 2 * std_4h
    
    # 4-hour Volume confirmation: current > 1.5x median of last 20 bars
    vol_median = pd.Series(volume).rolling(window=20, min_periods=1).median()
    vol_threshold = 1.5 * vol_median
    
    signals = np.zeros(n)
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(vol_threshold[i]) or np.isnan(ema_50_aligned[i]) or np.isnan(atr_1d_aligned[i]):
            continue
        
        # Volatility filter: avoid trading during extremely low volatility
        if atr_1d_aligned[i] < 0.5 * np.nanmedian(atr_1d_aligned[max(0, i-50):i+1]):
            signals[i] = signals[i-1] if i > 0 else 0.0
            continue
        
        # Trend filter: only long above daily EMA50, short below
        trend_filter = close[i] > ema_50_aligned[i]  # True for long bias, False for short bias
        
        # Long: close breaks above upper band + volume confirmation + trend filter
        if close[i] > upper[i] and volume[i] > vol_threshold[i] and trend_filter:
            signals[i] = 0.25
        
        # Short: close breaks below lower band + volume confirmation + trend filter
        elif close[i] < lower[i] and volume[i] > vol_threshold[i] and not trend_filter:
            signals[i] = -0.25
        
        # Exit: close crosses back inside bands (mean reversion)
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and close[i] < upper[i]) or
               (signals[i-1] == -0.25 and close[i] > lower[i]))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "4h_Bollinger_Breakout_Volume_TrendFilter"
timeframe = "4h"
leverage = 1.0