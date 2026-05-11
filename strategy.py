#!/usr/bin/env python3
name = "6h_VolumeWeighted_Keltner_Channel_1dTrend"
timeframe = "6h"
leverage = 1.0

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
    
    # Get 1d data for trend filter (using 1d EMA50)
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Keltner Channel parameters
    kc_mult = 2.0
    kc_length = 20
    
    # Calculate ATR for Keltner Channel
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=kc_length, min_periods=kc_length).mean().values
    
    # Calculate EMA for Keltner Channel middle line
    ema_middle = pd.Series(close).ewm(span=kc_length, min_periods=kc_length).mean().values
    
    # Calculate Keltner Channel bounds
    kc_upper = ema_middle + (kc_mult * atr)
    kc_lower = ema_middle - (kc_mult * atr)
    
    # Volume-weighted price for confirmation
    vwp = (close * volume) / np.where(volume != 0, volume, 1)
    vwp_ma = pd.Series(vwp).rolling(window=10, min_periods=10).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(kc_upper[i]) or np.isnan(kc_lower[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(vwp_ma[i]) or
            np.isnan(close[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above upper KC AND above 1d EMA50 AND volume-weighted price rising
            if close[i] > kc_upper[i] and close[i] > ema_1d_aligned[i] and vwp[i] > vwp_ma[i]:
                signals[i] = 0.25
                position = 1
            # Short: price below lower KC AND below 1d EMA50 AND volume-weighted price falling
            elif close[i] < kc_lower[i] and close[i] < ema_1d_aligned[i] and vwp[i] < vwp_ma[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price below middle KC OR below 1d EMA50
            if close[i] < ema_middle[i] or close[i] < ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: price above middle KC OR above 1d EMA50
            if close[i] > ema_middle[i] or close[i] > ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals