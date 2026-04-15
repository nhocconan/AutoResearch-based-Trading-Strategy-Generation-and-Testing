#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily ATR (14-period) for volatility filter
    daily = get_htf_data(prices, '1d')
    high_d = daily['high'].values
    low_d = daily['low'].values
    close_d = daily['close'].values
    tr1 = np.maximum(high_d[1:] - low_d[1:], np.abs(high_d[1:] - close_d[:-1]))
    tr2 = np.maximum(np.abs(low_d[1:] - close_d[:-1]), tr1)
    tr = np.concatenate([[np.nan], tr2])
    atr_14d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14d_aligned = align_htf_to_ltf(prices, daily, atr_14d)
    
    # Daily 20-period EMA for trend filter
    ema_20d = pd.Series(close_d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20d_aligned = align_htf_to_ltf(prices, daily, ema_20d)
    
    # Volume confirmation: current > 1.5x median of last 30 bars
    vol_median = pd.Series(volume).rolling(window=30, min_periods=30).median()
    vol_threshold = 1.5 * vol_median
    
    # Daily 50-period EMA for trend confirmation
    ema_50d = pd.Series(close_d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50d_aligned = align_htf_to_ltf(prices, daily, ema_50d)
    
    signals = np.zeros(n)
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_14d_aligned[i]) or np.isnan(ema_20d_aligned[i]) or 
            np.isnan(ema_50d_aligned[i]) or np.isnan(vol_threshold[i])):
            continue
        
        # Volatility filter: avoid extremes (0.5x to 3.0x of median ATR)
        atr_median = pd.Series(atr_14d_aligned).rolling(window=50, min_periods=50).median()
        vol_filter = (atr_14d_aligned[i] > 0.5 * atr_median[i]) and (atr_14d_aligned[i] < 3.0 * atr_median[i])
        
        # Trend filter: EMA20 > EMA50 for long, EMA20 < EMA50 for short
        trend_up = ema_20d_aligned[i] > ema_50d_aligned[i]
        trend_down = ema_20d_aligned[i] < ema_50d_aligned[i]
        
        # Long: Price above daily EMA20 + volume spike + volatility filter + trend up
        if (close[i] > ema_20d_aligned[i] and 
            volume[i] > vol_threshold[i] and 
            vol_filter and 
            trend_up):
            signals[i] = 0.25
        
        # Short: Price below daily EMA20 + volume spike + volatility filter + trend down
        elif (close[i] < ema_20d_aligned[i] and 
              volume[i] > vol_threshold[i] and 
              vol_filter and 
              trend_down):
            signals[i] = -0.25
        
        # Exit: price crosses back below/above daily EMA20
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and close[i] < ema_20d_aligned[i]) or
               (signals[i-1] == -0.25 and close[i] > ema_20d_aligned[i]))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "12h_DailyEMA20_50_Vol1.5x_ATR14dFilter"
timeframe = "12h"
leverage = 1.0