#!/usr/bin/env python3
"""
Hypothesis: 6h Volume-Weighted Average Price (VWAP) deviation with 1w trend filter.
Long when price > 6h VWAP AND 1w EMA50 rising AND volume > 1.5x average.
Short when price < 6h VWAP AND 1w EMA50 falling AND volume > 1.5x average.
Exit when price crosses VWAP or volume drops below average.
VWAP acts as dynamic support/resistance. Volume confirmation ensures conviction.
1w EMA50 filter ensures trading with weekly trend, reducing whipsaws in ranging markets.
Designed for 6h timeframe targeting 50-150 total trades over 4 years with low frequency.
Works in both bull and bear markets by only taking trades aligned with weekly trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data for EMA50 trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate EMA50 on 1w data
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w EMA50 to 6h timeframe
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate VWAP on 6h data (typical price * volume)
    typical_price = (high + low + close) / 3.0
    vol_sum = pd.Series(volume).rolling(window=20, min_periods=20).sum().values
    tp_vol_sum = pd.Series(typical_price * volume).rolling(window=20, min_periods=20).sum().values
    vwap = tp_vol_sum / vol_sum
    
    # Volume average (20-period) on primary timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(vwap[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema50_val = ema50_1w_aligned[i]
        vwap_val = vwap[i]
        vol_ma_val = vol_ma[i]
        price = close[i]
        vol_current = volume[i]
        
        # Calculate 1w EMA50 slope (rising/falling)
        if i >= 101:
            ema50_prev = ema50_1w_aligned[i-1]
            ema50_rising = ema50_val > ema50_prev
            ema50_falling = ema50_val < ema50_prev
        else:
            ema50_rising = False
            ema50_falling = False
        
        if position == 0:
            # Long: Price > VWAP AND 1w EMA50 rising AND volume spike
            if (price > vwap_val and ema50_rising and vol_current > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: Price < VWAP AND 1w EMA50 falling AND volume spike
            elif (price < vwap_val and ema50_falling and vol_current > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Price crosses below VWAP OR volume drops below average
                if (price < vwap_val or vol_current < vol_ma_val):
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price crosses above VWAP OR volume drops below average
                if (price > vwap_val or vol_current < vol_ma_val):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_VWAP_1wEMA50_Volume"
timeframe = "6h"
leverage = 1.0