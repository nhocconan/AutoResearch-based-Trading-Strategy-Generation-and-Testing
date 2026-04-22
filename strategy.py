#!/usr/bin/env python3
"""
Hypothesis: 4-hour Volume-Weighted Average Price (VWAP) Reversion with 12-hour Trend Filter.
Long when price crosses below VWAP during 12-hour uptrend with volume confirmation.
Short when price crosses above VWAP during 12-hour downtrend with volume confirmation.
Exit when price returns to VWAP or trend reverses.
Designed for low-to-moderate trade frequency by requiring trend alignment and volume confirmation.
Works in both bull and bear markets by following the 12-hour trend.
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
    
    # Typical price for VWAP calculation
    typical_price = (high + low + close) / 3.0
    
    # Cumulative volume and cumulative typical price * volume for VWAP
    cum_vol = np.cumsum(volume)
    cum_vol_price = np.cumsum(typical_price * volume)
    
    # VWAP calculation with reset at session start (simplified: using cumulative)
    vwap = cum_vol_price / cum_vol
    # Handle division by zero at start
    vwap = np.where(cum_vol == 0, typical_price, vwap)
    
    # Load 12-hour data for trend filter - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # 20-period EMA on 12h close for trend
    close_12h = df_12h['close'].values
    ema20_12h = pd.Series(close_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema20_12h)
    
    # Volume confirmation: current volume > 1.3x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(vwap[i]) or np.isnan(ema20_12h_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.3 * vol_ma_20[i]
        
        # Price-VWAP cross signals
        price_above_vwap = close[i] > vwap[i]
        price_below_vwap = close[i] < vwap[i]
        was_above_vwap = i > 0 and close[i-1] > vwap[i-1]
        was_below_vwap = i > 0 and close[i-1] < vwap[i-1]
        
        cross_below = price_below_vwap and was_above_vwap  # crossed below VWAP
        cross_above = price_above_vwap and was_below_vwap  # crossed above VWAP
        
        if position == 0:
            # Long: price crosses below VWAP + 12h uptrend + volume confirmation
            if cross_below and ema20_12h_aligned[i] > ema20_12h_aligned[i-1] and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: price crosses above VWAP + 12h downtrend + volume confirmation
            elif cross_above and ema20_12h_aligned[i] < ema20_12h_aligned[i-1] and vol_confirm:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price returns to VWAP or trend reverses
            exit_signal = False
            
            if position == 1:
                # Exit long: price crosses above VWAP or 12h trend turns down
                if cross_above or ema20_12h_aligned[i] < ema20_12h_aligned[i-1]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price crosses below VWAP or 12h trend turns up
                if cross_below or ema20_12h_aligned[i] > ema20_12h_aligned[i-1]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_VWAP_Reversion_12hTrend_Volume"
timeframe = "4h"
leverage = 1.0