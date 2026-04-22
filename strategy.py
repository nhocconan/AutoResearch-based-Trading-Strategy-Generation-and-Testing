#!/usr/bin/env python3
"""
Hypothesis: 4-hour Volume-Weighted Average Price (VWAP) with 1-day trend filter.
Long when price crosses above VWAP, 1-day EMA50 is rising, and volume is above average.
Short when price crosses below VWAP, 1-day EMA50 is falling, and volume is above average.
Exit when price crosses back across VWAP or volume dries up.
VWAP provides dynamic intraday support/resistance; 1-day EMA50 filters higher timeframe trend.
Designed for low trade frequency by requiring volume confirmation and trend alignment.
Works in both bull and bear markets by following daily trend while using 4h VWAP for entries.
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
    
    # Load 1-day data for EMA50 trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # VWAP calculation (typical price * volume) / cumulative volume
    typical_price = (high + low + close) / 3.0
    pv = typical_price * volume
    cum_pv = np.nancumsum(pv)
    cum_vol = np.nancumsum(volume)
    vwap = np.divide(cum_pv, cum_vol, out=np.zeros_like(cum_pv), where=cum_vol!=0)
    
    # Average volume (20-period) for confirmation
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after enough data for VWAP and volume average
        # Skip if data not ready
        if (np.isnan(vwap[i]) or np.isnan(avg_volume[i]) or 
            np.isnan(ema50_1d_aligned[i]) or volume[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price crosses above VWAP, volume above average, and 1-day EMA50 rising
            if (close[i] > vwap[i] and close[i-1] <= vwap[i-1] and 
                volume[i] > avg_volume[i] and 
                ema50_1d_aligned[i] > ema50_1d_aligned[i-1]):
                signals[i] = 0.25
                position = 1
            # Short: Price crosses below VWAP, volume above average, and 1-day EMA50 falling
            elif (close[i] < vwap[i] and close[i-1] >= vwap[i-1] and 
                  volume[i] > avg_volume[i] and 
                  ema50_1d_aligned[i] < ema50_1d_aligned[i-1]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Price crosses back below VWAP OR volume drops below average
                if (close[i] < vwap[i] and close[i-1] >= vwap[i-1]) or \
                   (volume[i] < avg_volume[i] * 0.5):  # Volume drops significantly
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price crosses back above VWAP OR volume drops below average
                if (close[i] > vwap[i] and close[i-1] <= vwap[i-1]) or \
                   (volume[i] < avg_volume[i] * 0.5):  # Volume drops significantly
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_VWAP_1dEMA50_Trend_Volume"
timeframe = "4h"
leverage = 1.0