#!/usr/bin/env python3

"""
Hypothesis: 4-hour Donchian breakout with 1-day ATR filter and volume confirmation.
Donchian channels provide clear breakout levels with objective entry/exit.
ATR filter ensures volatility is sufficient to avoid whipsaws in choppy markets.
Volume confirmation validates institutional participation at breakout points.
Target: 20-50 trades/year per symbol (80-200 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channels (20-period) on primary timeframe
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate ATR (14-period) for volatility filter
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Load 1d data for ATR filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d ATR (10-period) for volatility regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1_1d = high_1d[1:] - low_1d[1:]
    tr2_1d = np.abs(high_1d[1:] - close_1d[:-1])
    tr3_1d = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.concatenate([[np.nan], np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))])
    atr_10_1d = pd.Series(tr_1d).ewm(span=10, adjust=False, min_periods=10).mean().values
    atr_10_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_10_1d)
    
    # Calculate 4h volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(atr[i]) or np.isnan(atr_10_1d_aligned[i]) or
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volatility filter: current 4h ATR > 0.5 * 1d ATR (avoid low volatility chop)
        vol_filter = atr[i] > 0.5 * atr_10_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above Donchian high + volume spike + volatility filter
            if (close[i] > donchian_high[i] and
                volume[i] > 1.5 * vol_avg_20[i] and
                vol_filter):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low + volume spike + volatility filter
            elif (close[i] < donchian_low[i] and
                  volume[i] > 1.5 * vol_avg_20[i] and
                  vol_filter):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price returns to opposite Donchian level or volatility drops
            exit_signal = False
            
            if position == 1:
                # Exit long: price breaks below Donchian low
                if close[i] < donchian_low[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price breaks above Donchian high
                if close[i] > donchian_high[i]:
                    exit_signal = True
            
            # Additional exit: volatility drops too low
            if atr[i] < 0.3 * atr_10_1d_aligned[i]:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Donchian_Breakout_1dATR_Volume"
timeframe = "4h"
leverage = 1.0