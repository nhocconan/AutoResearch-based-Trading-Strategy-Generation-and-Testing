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
    
    # Weekly high/low for 1-week lookback (current week not yet closed)
    weekly = get_htf_data(prices, '1w')
    high_w = weekly['high'].values
    low_w = weekly['low'].values
    
    # Weekly highest high and lowest low over last 4 completed weeks (20 periods)
    highest_high = np.full(len(high_w), np.nan)
    lowest_low = np.full(len(low_w), np.nan)
    for i in range(4, len(high_w)):
        highest_high[i] = np.max(high_w[i-4:i])
        lowest_low[i] = np.min(low_w[i-4:i])
    weekly_high = align_htf_to_ltf(prices, weekly, highest_high)
    weekly_low = align_htf_to_ltf(prices, weekly, lowest_low)
    
    # Daily ATR(14) for volatility filter
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
    
    # Volume filter: 1.5x 20-day median volume
    vol_median = pd.Series(volume).rolling(window=20, min_periods=20).median()
    vol_threshold = 1.5 * vol_median
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(weekly_high[i]) or np.isnan(weekly_low[i]) or 
            np.isnan(atr_14d_aligned[i]) or np.isnan(vol_threshold[i])):
            continue
        
        # Long: Close above weekly high (breakout) + volume spike + volatility filter
        if (close[i] > weekly_high[i] and 
            volume[i] > vol_threshold[i] and
            atr_14d_aligned[i] > 0):
            signals[i] = 0.25
        
        # Short: Close below weekly low (breakdown) + volume spike + volatility filter
        elif (close[i] < weekly_low[i] and 
              volume[i] > vol_threshold[i] and
              atr_14d_aligned[i] > 0):
            signals[i] = -0.25
        
        # Exit: price returns to midpoint of weekly range
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and close[i] < (weekly_high[i] + weekly_low[i]) / 2) or
               (signals[i-1] == -0.25 and close[i] > (weekly_high[i] + weekly_low[i]) / 2))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "1d_WeeklyBreakout_Vol1.5x_ATR14dFilter"
timeframe = "1d"
leverage = 1.0