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
    
    # Calculate 14-period ATR for volatility filter
    tr = np.maximum(high[1:] - low[1:], np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
    tr = np.concatenate([[np.nan], tr])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Daily median volume filter (from daily data)
    df_1d = get_htf_data(prices, '1d')
    daily_vol = df_1d['volume'].values
    daily_vol_median = pd.Series(daily_vol).rolling(window=10, min_periods=10).median().values
    daily_vol_median_aligned = align_htf_to_ltf(prices, df_1d, daily_vol_median)
    
    # 6-hour price change momentum (close - open over 6h)
    price_change = close - prices['open'].values
    price_change_ma = pd.Series(price_change).rolling(window=6, min_periods=6).mean().values
    
    signals = np.zeros(n)
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if np.isnan(atr[i]) or np.isnan(daily_vol_median_aligned[i]) or np.isnan(price_change_ma[i]):
            continue
        
        # Volume condition: current 6h volume > 1.5x daily median volume / 4 (approx 6h slice)
        vol_cond = volume[i] > 1.5 * (daily_vol_median_aligned[i] / 4)
        
        # Momentum condition: significant 6-bar price momentum
        mom_cond = np.abs(price_change_ma[i]) > 0.5 * atr[i]
        
        # Long: positive momentum + volume confirmation
        if price_change_ma[i] > 0 and vol_cond and mom_cond:
            signals[i] = 0.25
        
        # Short: negative momentum + volume confirmation
        elif price_change_ma[i] < 0 and vol_cond and mom_cond:
            signals[i] = -0.25
        
        # Exit: momentum fades or volume drops
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and price_change_ma[i] <= 0) or
               (signals[i-1] == -0.25 and price_change_ma[i] >= 0) or
               volume[i] <= 0.8 * (daily_vol_median_aligned[i] / 4))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "6h_Momentum_Volume_Filter"
timeframe = "6h"
leverage = 1.0