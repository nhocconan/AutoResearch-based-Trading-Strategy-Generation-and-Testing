#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_WickReversal_VolumeTrend"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1-week EMA50 for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Daily data for Wick Reversal levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily Wick Reversal levels
    n1d = len(close_1d)
    wick_reversal_high = np.full(n1d, np.nan)
    wick_reversal_low = np.full(n1d, np.nan)
    
    for i in range(1, n1d):
        PH = high_1d[i-1]
        PL = low_1d[i-1]
        PC = close_1d[i-1]
        
        # Wick Reversal levels: previous day high/low + 50% of range
        wick_reversal_high[i] = PH + 0.5 * (PH - PL)
        wick_reversal_low[i] = PL - 0.5 * (PH - PL)
    
    # Align Wick Reversal levels to 12h timeframe
    wick_reversal_high_aligned = align_htf_to_ltf(prices, df_1d, wick_reversal_high)
    wick_reversal_low_aligned = align_htf_to_ltf(prices, df_1d, wick_reversal_low)
    
    # Volume confirmation: current volume > 1.5x 24-period average
    vol_ma24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_confirmed = volume > (1.5 * vol_ma24)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(wick_reversal_high_aligned[i]) or np.isnan(wick_reversal_low_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_confirmed[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price closes below Wick Low with weekly uptrend + volume confirmation
            long_cond = (close[i] < wick_reversal_low_aligned[i] and 
                        ema_50_1w_aligned[i] > ema_50_1w_aligned[i-1] and
                        volume_confirmed[i])
            
            # Short: price closes above Wick High with weekly downtrend + volume confirmation
            short_cond = (close[i] > wick_reversal_high_aligned[i] and 
                         ema_50_1w_aligned[i] < ema_50_1w_aligned[i-1] and
                         volume_confirmed[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price closes above Wick High (mean reversion)
            if close[i] > wick_reversal_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes below Wick Low (mean reversion)
            if close[i] < wick_reversal_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals