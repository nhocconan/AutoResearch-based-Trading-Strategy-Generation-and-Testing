#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for indicators
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 14-day ATR for volatility filter
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 20-day high and low for breakout levels
    high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Calculate 50-day EMA for trend filter
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all daily data to 12h timeframe
    high_20_12h = align_htf_to_ltf(prices, df_1d, high_20)
    low_20_12h = align_htf_to_ltf(prices, df_1d, low_20)
    ema_50_12h = align_htf_to_ltf(prices, df_1d, ema_50)
    atr_14_12h = align_htf_to_ltf(prices, df_1d, atr_14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # wait for EMA50
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(high_20_12h[i]) or np.isnan(low_20_12h[i]) or 
            np.isnan(ema_50_12h[i]) or np.isnan(atr_14_12h[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter
        uptrend = close[i] > ema_50_12h[i]
        downtrend = close[i] < ema_50_12h[i]
        
        # Volatility filter: only trade when volatility is elevated
        vol_filter = atr_14_12h[i] > np.nanmean(atr_14_12h[max(0, i-50):i]) * 1.2
        
        if position == 0:
            # Long: price breaks above 20-day high with uptrend and elevated volatility
            if close[i] > high_20_12h[i] and uptrend and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 20-day low with downtrend and elevated volatility
            elif close[i] < low_20_12h[i] and downtrend and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price closes below 20-day low OR trend reverses
            if close[i] < low_20_12h[i] or not uptrend:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price closes above 20-day high OR trend reverses
            if close[i] > high_20_12h[i] or not downtrend:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_ATRVol_Breakout_EMA50_20D_v1"
timeframe = "12h"
leverage = 1.0