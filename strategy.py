#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1w EMA(50) trend filter and volume confirmation
# Enter long when: price > Donchian upper(20), price > 1w EMA(50), volume > 1.5x avg
# Enter short when: price < Donchian lower(20), price < 1w EMA(50), volume > 1.5x avg
# Exit when: opposite Donchian band touched or trailing stop at 2*ATR(14)
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe
# Works in bull/bear via trend filter and volatility-based stops

name = "12h_donchian20_1w_ema_vol_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w EMA(50) for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # Donchian channels (20-period) on 12h
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * volume_ma
    
    # ATR(14) for trailing stop
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    for i in range(20, n):  # Wait for indicators to stabilize
        # Skip if required data not available
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_threshold[i]) or
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit conditions
            exit_condition = False
            # 1. Price touches lower Donchian band
            if low[i] <= low_20[i]:
                exit_condition = True
            # 2. Trailing stop: price drops 2*ATR from highest high since entry
            elif i > 0 and entry_price > 0:
                highest_since_entry = np.max(high[entry_idx:i+1]) if 'entry_idx' in locals() else high[i]
                if low[i] <= highest_since_entry - 2.0 * atr[i]:
                    exit_condition = True
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit conditions
            exit_condition = False
            # 1. Price touches upper Donchian band
            if high[i] >= high_20[i]:
                exit_condition = True
            # 2. Trailing stop: price rises 2*ATR from lowest low since entry
            elif i > 0 and entry_price > 0:
                lowest_since_entry = np.min(low[entry_idx:i+1]) if 'entry_idx' in locals() else low[i]
                if high[i] >= lowest_since_entry + 2.0 * atr[i]:
                    exit_condition = True
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + trend filter + volume
            if volume[i] > volume_threshold[i]:
                # Long: price breaks above upper Donchian band and above 1w EMA(50)
                if high[i] > high_20[i] and close[i] > ema_50_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                    entry_idx = i
                # Short: price breaks below lower Donchian band and below 1w EMA(50)
                elif low[i] < low_20[i] and close[i] < ema_50_aligned[i]:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
                    entry_idx = i
    
    return signals