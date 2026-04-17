#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with volume confirmation and ATR-based trailing stop.
Long when price breaks above 20-period high with volume > 1.5x average.
Short when price breaks below 20-period low with volume > 1.5x average.
Exit via ATR trailing stop (3x ATR from extreme) or opposite breakout.
Uses 1d EMA50 as trend filter: only long when price > EMA50, short when price < EMA50.
Target: 75-200 total trades over 4 years (19-50/year). Discrete sizing 0.25.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate ATR(14) for volatility and trailing stop
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[tr1[0]], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    long_stop = 0.0
    short_stop = 0.0
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or 
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_conf = volume_confirm[i]
        ema50 = ema50_1d_aligned[i]
        atr_val = atr[i]
        upper = highest_high[i]
        lower = lowest_low[i]
        
        if position == 0:
            # Long: price breaks above upper Donchian with volume confirmation and above EMA50
            if price > upper and vol_conf and price > ema50:
                signals[i] = 0.25
                position = 1
                long_stop = price - 3.0 * atr_val  # initial stop
            # Short: price breaks below lower Donchian with volume confirmation and below EMA50
            elif price < lower and vol_conf and price < ema50:
                signals[i] = -0.25
                position = -1
                short_stop = price + 3.0 * atr_val  # initial stop
        
        elif position == 1:
            # Update trailing stop for long
            long_stop = max(long_stop, price - 3.0 * atr_val)
            
            # Exit long: price hits stop or opposite breakout (below lower Donchian)
            if price <= long_stop or price < lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Update trailing stop for short
            short_stop = min(short_stop, price + 3.0 * atr_val)
            
            # Exit short: price hits stop or opposite breakout (above upper Donchian)
            if price >= short_stop or price > upper:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_VolumeConfirm_EMA50Trend_ATRTrail"
timeframe = "4h"
leverage = 1.0