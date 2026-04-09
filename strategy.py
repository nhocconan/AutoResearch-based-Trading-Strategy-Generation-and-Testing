#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using Donchian(20) breakout with volume confirmation and ATR stoploss
# Donchian breakout captures momentum; volume confirmation filters false breakouts
# ATR-based stoploss manages risk. Works in both bull and bear markets by following price channels
# Target: 75-200 total trades over 4 years (19-50/year) to avoid fee drag

name = "4h_donchian_breakout_volume_atr_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channels (20-period)
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Calculate ATR(14) for stoploss
    tr1 = np.abs(high[1:] - low[:-1])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    def wilders_smoothing(values, period):
        if len(values) < period:
            return np.full(len(values), np.nan)
        alpha = 1.0 / period
        result = np.full(len(values), np.nan)
        result[period-1] = np.nanmean(values[:period])
        for i in range(period, len(values)):
            result[i] = alpha * values[i] + (1 - alpha) * result[i-1]
        return result
    
    atr = wilders_smoothing(tr, 14)
    
    # Volume confirmation: 20-period volume SMA
    volume_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(lookback, n):
        # Skip if any required data is invalid
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(atr[i]) or np.isnan(volume_sma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit long if price hits ATR-based stoploss or breaks below Donchian low
            if close[i] <= highest_high[i-1] - 2.0 * atr[i] or close[i] < lowest_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit short if price hits ATR-based stoploss or breaks above Donchian high
            if close[i] >= lowest_low[i-1] + 2.0 * atr[i] or close[i] > highest_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long on Donchian high breakout with volume confirmation
            if (high[i] > highest_high[i-1] and 
                volume[i] > volume_sma[i]):
                position = 1
                signals[i] = 0.25
            # Enter short on Donchian low breakdown with volume confirmation
            elif (low[i] < lowest_low[i-1] and 
                  volume[i] > volume_sma[i]):
                position = -1
                signals[i] = -0.25
    
    return signals