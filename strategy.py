#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1-day data for ATR calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range for ATR(14)
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # ATR with Wilder's smoothing
    atr = np.zeros_like(tr)
    atr[0] = tr[0]
    for i in range(1, len(tr)):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # ATR ratio: ATR(7)/ATR(14)
    atr7 = np.zeros_like(tr)
    atr7[0] = tr[0]
    for i in range(1, len(tr)):
        atr7[i] = (atr7[i-1] * 6 + tr[i]) / 7
    
    atr_ratio = np.divide(atr7, atr, out=np.zeros_like(atr7), where=atr!=0)
    
    # Volume spike filter (20-period on 1h)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.5 * vol_ma20  # Require 2.5x volume for confirmation
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Align indicators to 1-hour timeframe
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike)  # Actually vol_spike is 1h, but keeping for consistency
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(200, n):  # Start after warmup
        # Skip if data not ready or outside session
        if (np.isnan(atr_ratio_aligned[i]) or 
            np.isnan(vol_ma20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Volatility contraction (low ATR ratio) + volume expansion
            if (atr_ratio_aligned[i] < 0.6 and vol_spike[i]):
                signals[i] = 0.20
                position = 1
            # Short: Volatility contraction + volume expansion
            elif (atr_ratio_aligned[i] < 0.6 and vol_spike[i]):
                signals[i] = -0.20
                position = -1
        else:
            # Exit: Volatility expansion (high ATR ratio) or session end approaching
            if position == 1:
                if (atr_ratio_aligned[i] > 0.8 or 
                    (i < n-1 and not in_session[i+1])):  # Exit before session ends
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
            else:  # position == -1
                if (atr_ratio_aligned[i] > 0.8 or 
                    (i < n-1 and not in_session[i+1])):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
    
    return signals

name = "1h_ATR_Ratio_Vol_Spike_Session"
timeframe = "1h"
leverage = 1.0