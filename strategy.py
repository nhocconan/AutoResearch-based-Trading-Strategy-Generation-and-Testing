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
    
    # Weekly high/low for trend context
    weekly = get_htf_data(prices, '1w')
    high_w = weekly['high'].values
    low_w = weekly['low'].values
    
    # Weekly high/low - only update on weekly bar close
    weekly_high = np.maximum.accumulate(high_w)
    weekly_low = np.minimum.accumulate(low_w)
    weekly_high_aligned = align_htf_to_ltf(prices, weekly, weekly_high)
    weekly_low_aligned = align_htf_to_ltf(prices, weekly, weekly_low)
    
    # Daily ATR(14) for volatility normalization
    daily = get_htf_data(prices, '1d')
    high_d = daily['high'].values
    low_d = daily['low'].values
    close_d = daily['close'].values
    
    tr1 = high_d[1:] - low_d[1:]
    tr2 = np.abs(high_d[1:] - close_d[:-1])
    tr3 = np.abs(low_d[1:] - close_d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14d_aligned = align_htf_to_ltf(prices, daily, atr_14d)
    
    # Daily ATR median for regime filter
    atr_median = pd.Series(atr_14d_aligned).rolling(window=50, min_periods=50).median()
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(weekly_high_aligned[i]) or np.isnan(weekly_low_aligned[i]) or
            np.isnan(atr_14d_aligned[i]) or np.isnan(atr_median[i])):
            continue
        
        # Volatility regime: avoid extreme volatility (>3x median ATR)
        vol_regime = atr_14d_aligned[i] < 3.0 * atr_median[i]
        
        # Long: Price breaks above weekly high with volume confirmation
        long_breakout = (close[i] > weekly_high_aligned[i]) and (volume[i] > 1.5 * np.median(volume[max(0,i-20):i+1]))
        
        # Short: Price breaks below weekly low with volume confirmation
        short_breakout = (close[i] < weekly_low_aligned[i]) and (volume[i] > 1.5 * np.median(volume[max(0,i-20):i+1]))
        
        if long_breakout and vol_regime:
            signals[i] = 0.25
        elif short_breakout and vol_regime:
            signals[i] = -0.25
        else:
            signals[i] = signals[i-1] if i > 0 else 0.0
    
    return signals

name = "6h_WeeklyBreakout_VolumeFilter"
timeframe = "6h"
leverage = 1.0