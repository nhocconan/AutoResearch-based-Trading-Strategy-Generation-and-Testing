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
    
    # 12h timeframe
    daily12h = get_htf_data(prices, '12h')
    close_12h = daily12h['close'].values
    high_12h = daily12h['high'].values
    low_12h = daily12h['low'].values
    
    # 12h EMA200 for trend filter
    close_12h_series = pd.Series(close_12h)
    ema200_12h = close_12h_series.ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_12h_aligned = align_htf_to_ltf(prices, daily12h, ema200_12h)
    
    # 12h ATR(14) for volatility filter
    tr1 = np.maximum(high_12h[1:] - low_12h[1:], np.abs(high_12h[1:] - close_12h[:-1]))
    tr2 = np.maximum(np.abs(low_12h[1:] - close_12h[:-1]), tr1)
    tr = np.concatenate([[np.nan], tr2])
    atr_14_12h = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_12h_aligned = align_htf_to_ltf(prices, daily12h, atr_14_12h)
    
    # 12h Donchian channels (20-period)
    donch_high_12h = np.full(len(close_12h), np.nan)
    donch_low_12h = np.full(len(close_12h), np.nan)
    for i in range(20, len(close_12h)):
        donch_high_12h[i] = np.max(high_12h[i-20:i])
        donch_low_12h[i] = np.min(low_12h[i-20:i])
    donch_high_12h_aligned = align_htf_to_ltf(prices, daily12h, donch_high_12h)
    donch_low_12h_aligned = align_htf_to_ltf(prices, daily12h, donch_low_12h)
    
    # 6h volume filter: 2.0x median of last 20 bars
    vol_median = pd.Series(volume).rolling(window=20, min_periods=20).median()
    vol_threshold = 2.0 * vol_median
    
    signals = np.zeros(n)
    
    for i in range(200, n):
        # Skip if any required data is NaN
        if (np.isnan(ema200_12h_aligned[i]) or np.isnan(atr_14_12h_aligned[i]) or
            np.isnan(donch_high_12h_aligned[i]) or np.isnan(donch_low_12h_aligned[i]) or
            np.isnan(vol_threshold[i])):
            continue
        
        # Long: Price above EMA200 + breaks above 12h Donchian high + volume spike
        if (close[i] > ema200_12h_aligned[i] and 
            close[i] > donch_high_12h_aligned[i] and 
            volume[i] > vol_threshold[i]):
            signals[i] = 0.25
        
        # Short: Price below EMA200 + breaks below 12h Donchian low + volume spike
        elif (close[i] < ema200_12h_aligned[i] and 
              close[i] < donch_low_12h_aligned[i] and 
              volume[i] > vol_threshold[i]):
            signals[i] = -0.25
        
        # Exit: price returns to middle of 12h Donchian channel
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and close[i] < (donch_high_12h_aligned[i] + donch_low_12h_aligned[i]) / 2) or
               (signals[i-1] == -0.25 and close[i] > (donch_high_12h_aligned[i] + donch_low_12h_aligned[i]) / 2))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "6h_12h_EMA200_Donchian20_Vol2x_Trend"
timeframe = "6h"
leverage = 1.0