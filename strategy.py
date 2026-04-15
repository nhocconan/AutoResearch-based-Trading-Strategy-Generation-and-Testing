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
    
    # 1d Close for Donchian channels
    daily = get_htf_data(prices, '1d')
    close_d = daily['close'].values
    high_d = daily['high'].values
    low_d = daily['low'].values
    
    # Daily Donchian(20) for breakout signals
    donchian_high = pd.Series(high_d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_d).rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, daily, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, daily, donchian_low)
    
    # Daily ATR(14) for volatility filter
    tr1 = np.maximum(high_d[1:] - low_d[1:], np.abs(high_d[1:] - close_d[:-1]))
    tr2 = np.maximum(np.abs(low_d[1:] - close_d[:-1]), tr1)
    tr = np.concatenate([[np.nan], tr2])
    atr_14d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14d_aligned = align_htf_to_ltf(prices, daily, atr_14d)
    
    # 1w High/Low for trend filter (longer-term context)
    weekly = get_htf_data(prices, '1w')
    close_w = weekly['close'].values
    high_w = weekly['high'].values
    low_w = weekly['low'].values
    
    # Weekly EMA(50) for trend direction
    ema_50w = pd.Series(close_w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50w_aligned = align_htf_to_ltf(prices, weekly, ema_50w)
    
    # Volume threshold: 1.8x median of last 20 bars
    vol_median = pd.Series(volume).rolling(window=20, min_periods=20).median()
    vol_threshold = 1.8 * vol_median
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(atr_14d_aligned[i]) or np.isnan(ema_50w_aligned[i]) or
            np.isnan(vol_threshold[i])):
            continue
        
        # Trend filter: price above/below weekly EMA50
        uptrend = close[i] > ema_50w_aligned[i]
        downtrend = close[i] < ema_50w_aligned[i]
        
        # Breakout signals with volume confirmation
        long_breakout = (close[i] > donchian_high_aligned[i]) and (volume[i] > vol_threshold[i])
        short_breakout = (close[i] < donchian_low_aligned[i]) and (volume[i] > vol_threshold[i])
        
        # Entry logic
        if long_breakout and uptrend:
            signals[i] = 0.25
        elif short_breakout and downtrend:
            signals[i] = -0.25
        # Exit: opposite breakout or loss of trend
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and (close[i] < donchian_low_aligned[i] or not uptrend)) or
               (signals[i-1] == -0.25 and (close[i] > donchian_high_aligned[i] or not downtrend)))):
            signals[i] = 0.0
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "4h_Donchian20_WeeklyEMA50_Vol1.8x"
timeframe = "4h"
leverage = 1.0