#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation
# Uses 1d EMA50 for trend filter to capture multi-day momentum
# Entry logic: Break above 20-period Donchian high with volume spike in uptrend (close > 1d EMA50) for long
#              Break below 20-period Donchian low with volume spike in downtrend (close < 1d EMA50) for short
# Works in both bull and bear markets by trading with the 1d trend
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe
# Discrete sizing 0.25 balances profit potential and fee drag

name = "4h_Donchian20_Breakout_1dEMA50_Volume"
timeframe = "4h"
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
    
    # Calculate 4h Donchian channels (20-period)
    # Using rolling window on 4h data directly
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Session filter: 08-20 UTC (reduces noise trades)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = 100
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(high_roll[i]) or np.isnan(low_roll[i])):
            signals[i] = 0.0
            continue
        
        # Apply session filter
        if not in_session[i]:
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Break above Donchian high AND price > 1d EMA50 (uptrend) AND volume spike
            if (close[i] > high_roll[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: Break below Donchian low AND price < 1d EMA50 (downtrend) AND volume spike
            elif (close[i] < low_roll[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Close below 1d EMA50 (trend change) OR break below Donchian low (reversal)
            if (close[i] < ema_50_1d_aligned[i] or 
                close[i] < low_roll[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Close above 1d EMA50 (trend change) OR break above Donchian high (reversal)
            if (close[i] > ema_50_1d_aligned[i] or 
                close[i] > high_roll[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals