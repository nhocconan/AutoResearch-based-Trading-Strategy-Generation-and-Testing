#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) Breakout with 1d EMA50 Trend Filter and Volume Confirmation
# Uses daily EMA50 for trend filter and Donchian channel breakouts for entries
# Entry logic: Break above 20-period high with volume spike in uptrend (price > 1d EMA50) for long
#              Break below 20-period low with volume spike in downtrend (price < 1d EMA50) for short
# Exit logic: Close below 1d EMA50 (trend change) or opposite Donchian breakout
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe
# Discrete sizing 0.25 balances profit potential and fee drag
# Works in both bull and bear markets by trading with the daily trend

name = "12h_Donchian20_Breakout_1dEMA50_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Donchian channels (20-period) on primary timeframe
    high_ma = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_ma = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = 100
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(high_ma[i]) or np.isnan(low_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Break above 20-period high AND price > 1d EMA50 (uptrend) AND volume spike
            if (close[i] > high_ma[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: Break below 20-period low AND price < 1d EMA50 (downtrend) AND volume spike
            elif (close[i] < low_ma[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Close below 1d EMA50 (trend change) OR break below 20-period low (reversal)
            if (close[i] < ema_50_1d_aligned[i] or 
                close[i] < low_ma[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Close above 1d EMA50 (trend change) OR break above 20-period high (reversal)
            if (close[i] > ema_50_1d_aligned[i] or 
                close[i] > high_ma[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals