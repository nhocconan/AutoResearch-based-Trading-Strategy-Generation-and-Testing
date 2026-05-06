#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 12h Supertrend for trend direction, 6h ATR-based breakout with volume confirmation
# Supertrend on 12h filters trend direction to avoid counter-trend trades
# Breakout above/below ATR(14) multiplier from open with volume > 1.5x 20-period average signals momentum
# Works in bull/bear: Supertrend adapts to trend, breakouts capture momentum in trend direction
# Target: 60-120 total trades over 4 years (15-30/year) with 0.25 position sizing

name = "6h_Supertrend_ATRBreakout_Volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_price = prices['open'].values
    
    # Calculate 12h Supertrend ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 15:
        return np.zeros(n)
    
    # Supertrend parameters
    atr_period = 10
    multiplier = 3.0
    
    # Calculate ATR for 12h
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    atr = np.zeros_like(tr)
    atr[atr_period-1] = np.mean(tr[:atr_period])
    for i in range(atr_period, len(tr)):
        atr[i] = (atr[i-1] * (atr_period-1) + tr[i]) / atr_period
    
    # Calculate basic upper and lower bands
    basic_ub = (high_12h + low_12h) / 2 + multiplier * atr
    basic_lb = (high_12h + low_12h) / 2 - multiplier * atr
    
    # Initialize final bands
    final_ub = np.zeros_like(basic_ub)
    final_lb = np.zeros_like(basic_lb)
    supertrend = np.zeros_like(close_12h)
    direction = np.ones_like(close_12h)  # 1 for uptrend, -1 for downtrend
    
    # Calculate Supertrend
    for i in range(atr_period, len(close_12h)):
        if basic_ub[i] < final_ub[i-1] or close_12h[i-1] > final_ub[i-1]:
            final_ub[i] = basic_ub[i]
        else:
            final_ub[i] = final_ub[i-1]
            
        if basic_lb[i] > final_lb[i-1] or close_12h[i-1] < final_lb[i-1]:
            final_lb[i] = basic_lb[i]
        else:
            final_lb[i] = final_lb[i-1]
        
        if supertrend[i-1] == final_ub[i-1]:
            if close_12h[i] <= final_ub[i]:
                supertrend[i] = final_ub[i]
            else:
                supertrend[i] = final_lb[i]
                direction[i] = -1
        else:
            if close_12h[i] >= final_lb[i]:
                supertrend[i] = final_lb[i]
            else:
                supertrend[i] = final_ub[i]
                direction[i] = 1
    
    # Align Supertrend to 6h timeframe (direction: 1=uptrend, -1=downtrend)
    supertrend_direction = direction
    supertrend_dir_aligned = align_htf_to_ltf(prices, df_12h, supertrend_direction)
    
    # ATR-based breakout on 6h
    atr_period_6h = 14
    atr_multiplier = 1.5
    
    tr1_6h = high - low
    tr2_6h = np.abs(high - np.roll(close, 1))
    tr3_6h = np.abs(low - np.roll(close, 1))
    tr_6h = np.maximum(tr1_6h, np.maximum(tr2_6h, tr3_6h))
    tr_6h[0] = tr1_6h[0]
    
    atr_6h = np.zeros_like(tr_6h)
    atr_6h[atr_period_6h-1] = np.mean(tr_6h[:atr_period_6h])
    for i in range(atr_period_6h, len(tr_6h)):
        atr_6h[i] = (atr_6h[i-1] * (atr_period_6h-1) + tr_6h[i]) / atr_period_6h
    
    # Calculate upper and lower breakout levels from open
    ub_6h = open_price + atr_multiplier * atr_6h
    lb_6h = open_price - atr_multiplier * atr_6h
    
    # Volume confirmation: >1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(supertrend_dir_aligned[i]) or np.isnan(ub_6h[i]) or np.isnan(lb_6h[i]) or
            np.isnan(volume_filter[i]) or np.isnan(atr_6h[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price breaks above upper band with volume confirmation and uptrend
            if close[i] > ub_6h[i] and volume_filter[i] and supertrend_dir_aligned[i] == 1:
                signals[i] = 0.25
                position = 1
            # Short breakout: price breaks below lower band with volume confirmation and downtrend
            elif close[i] < lb_6h[i] and volume_filter[i] and supertrend_dir_aligned[i] == -1:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below open (failed breakout) or opposite Supertrend signal
            if close[i] < open_price[i] or supertrend_dir_aligned[i] == -1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above open (failed breakdown) or opposite Supertrend signal
            if close[i] > open_price[i] or supertrend_dir_aligned[i] == 1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals