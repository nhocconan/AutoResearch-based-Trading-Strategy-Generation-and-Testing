#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly ATR(14) for volatility filter
    weekly = get_htf_data(prices, '1w')
    high_w = weekly['high'].values
    low_w = weekly['low'].values
    close_w = weekly['close'].values
    
    # True Range calculation
    tr1 = high_w[1:] - low_w[1:]
    tr2 = np.abs(high_w[1:] - close_w[:-1])
    tr3 = np.abs(low_w[1:] - close_w[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14w = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14w_aligned = align_htf_to_ltf(prices, weekly, atr_14w)
    
    # Weekly Donchian channels (20-period)
    donch_high_w = np.full(len(close_w), np.nan)
    donch_low_w = np.full(len(close_w), np.nan)
    for i in range(20, len(close_w)):
        donch_high_w[i] = np.max(high_w[i-20:i])
        donch_low_w[i] = np.min(low_w[i-20:i])
    donch_high_w_aligned = align_htf_to_ltf(prices, weekly, donch_high_w)
    donch_low_w_aligned = align_htf_to_ltf(prices, weekly, donch_low_w)
    
    # Volume threshold: 2.0x median of last 50 bars
    vol_median = pd.Series(volume).rolling(window=50, min_periods=50).median()
    vol_threshold = 2.0 * vol_median
    
    signals = np.zeros(n)
    
    for i in range(200, n):
        # Skip if any required data is NaN
        if (np.isnan(donch_high_w_aligned[i]) or np.isnan(donch_low_w_aligned[i]) or
            np.isnan(atr_14w_aligned[i]) or np.isnan(vol_threshold[i])):
            continue
        
        # Long: Price breaks above weekly Donchian high + volume spike + volatility filter
        if (close[i] > donch_high_w_aligned[i] and 
            volume[i] > vol_threshold[i] and
            atr_14w_aligned[i] > 0):
            signals[i] = 0.25
        
        # Short: Price breaks below weekly Donchian low + volume spike + volatility filter
        elif (close[i] < donch_low_w_aligned[i] and 
              volume[i] > vol_threshold[i] and
              atr_14w_aligned[i] > 0):
            signals[i] = -0.25
        
        # Exit: price returns to middle of weekly Donchian channel
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and close[i] < (donch_high_w_aligned[i] + donch_low_w_aligned[i]) / 2) or
               (signals[i-1] == -0.25 and close[i] > (donch_high_w_aligned[i] + donch_low_w_aligned[i]) / 2))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "12h_WeeklyDonchian20_Vol2.0x_ATR14wFilter_v1"
timeframe = "12h"
leverage = 1.0