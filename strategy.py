#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d EMA50 trend + volume confirmation
# - Primary signal: Donchian(20) breakout - long when price > 20-period high, short when < 20-period low
# - Trend filter: 1d EMA50 - price must be above EMA for longs, below for shorts (higher timeframe alignment)
# - Volume confirmation: 4h volume > 20-period median volume (avoid low-participation signals)
# - Position size: 0.25 (discrete level) to minimize fee churn
# - Stoploss: ATR-based - exit long if price < highest - 2*ATR, exit short if price > lowest + 2*ATR
# - Works in bull/bear: Donchian captures breakouts, EMA50 filter ensures alignment with higher timeframe trend

name = "4h_1d_donchian_ema_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Pre-compute 4h Donchian channels
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Donchian(20) channels
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # 4h volume regime: volume > 20-period median volume
    volume = prices['volume'].values
    median_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    volume_regime = volume > median_volume_20
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_high = 0.0  # highest high since entry for trailing stop
    entry_low = 0.0   # lowest low since entry for trailing stop
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_aligned[i]) or
            np.isnan(highest_high[i]) or
            np.isnan(lowest_low[i]) or
            np.isnan(atr[i]) or
            np.isnan(volume_regime[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Update highest high since entry
            if high[i] > entry_high:
                entry_high = high[i]
            # Exit: price < highest_high - 2*ATR (trailing stop) OR Donchian break down
            if close[i] < entry_high - 2.0 * atr[i] or close[i] < lowest_low[i]:
                position = 0
                entry_high = 0.0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            if low[i] < entry_low:
                entry_low = low[i]
            # Exit: price > lowest_low + 2*ATR (trailing stop) OR Donchian break up
            if close[i] > entry_low + 2.0 * atr[i] or close[i] > highest_high[i]:
                position = 0
                entry_low = 0.0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Donchian breakout with volume confirmation and 1d EMA50 filter
            # Long: price > highest_high AND volume regime AND price above 1d EMA50
            if (close[i] > highest_high[i] and 
                volume_regime[i] and 
                close[i] > ema_50_aligned[i]):
                position = 1
                entry_high = high[i]
                signals[i] = 0.25
            # Short: price < lowest_low AND volume regime AND price below 1d EMA50
            elif (close[i] < lowest_low[i] and 
                  volume_regime[i] and 
                  close[i] < ema_50_aligned[i]):
                position = -1
                entry_low = low[i]
                signals[i] = -0.25
    
    return signals