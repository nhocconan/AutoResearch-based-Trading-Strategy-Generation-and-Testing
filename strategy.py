#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d trend filter (EMA50) and volume confirmation
# Long when price breaks above 20-period high AND close > 1d EMA50 AND volume > 1.5x average
# Short when price breaks below 20-period low AND close < 1d EMA50 AND volume > 1.5x average
# Exit when price crosses opposite Donchian level or volume drops below average
# Uses 4h timeframe to balance trade frequency and signal quality, targeting 75-200 total trades over 4 years
# Works in both bull/bear markets by following trend with breakout confirmation

name = "4h_donchian20_1d_ema_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian Channel (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max()
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min()
    
    # 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    daily_close = df_1d['close'].values
    ema50 = pd.Series(daily_close).ewm(span=50, min_periods=50, adjust=False).mean()
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50.values)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_threshold = 1.5 * volume_ma.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if required data not available
        if np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(ema50_aligned[i]) or np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Exit conditions
        if position == 1:  # long position
            if close[i] < lowest_low[i] or volume[i] < volume_ma.values[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if close[i] > highest_high[i] or volume[i] < volume_ma.values[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with trend and volume confirmation
            # Long: break above 20-period high + above 1d EMA50 + volume confirmation
            if close[i] > highest_high[i] and close[i] > ema50_aligned[i] and volume[i] > volume_threshold[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below 20-period low + below 1d EMA50 + volume confirmation
            elif close[i] < lowest_low[i] and close[i] < ema50_aligned[i] and volume[i] > volume_threshold[i]:
                signals[i] = -0.25
                position = -1
    
    return signals