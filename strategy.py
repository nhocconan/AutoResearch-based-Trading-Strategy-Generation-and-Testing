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
    
    # 12h Donchian breakout with volume confirmation
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Donchian channels (20 periods)
    donchian_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Align to 6h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low)
    
    # Volume confirmation: current > 1.5x median of last 20 bars
    vol_median = pd.Series(volume).rolling(window=20, min_periods=1).median()
    vol_threshold = 1.5 * vol_median
    
    # Momentum filter: RSI(14) > 50 for long, < 50 for short
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(vol_threshold[i]) or np.isnan(rsi[i])):
            continue
        
        # Long: close breaks above Donchian high + volume + RSI > 50
        if close[i] > donchian_high_aligned[i] and volume[i] > vol_threshold[i] and rsi[i] > 50:
            signals[i] = 0.25
        
        # Short: close breaks below Donchian low + volume + RSI < 50
        elif close[i] < donchian_low_aligned[i] and volume[i] > vol_threshold[i] and rsi[i] < 50:
            signals[i] = -0.25
        
        # Exit: close crosses back inside Donchian channels (mean reversion)
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and close[i] < donchian_high_aligned[i]) or
               (signals[i-1] == -0.25 and close[i] > donchian_low_aligned[i]))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "6h_Donchian_Breakout_Volume_RSI"
timeframe = "6h"
leverage = 1.0