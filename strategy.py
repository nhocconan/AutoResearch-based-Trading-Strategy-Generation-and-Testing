#!/usr/bin/env python3
# 12h_1d_1w_camarilla_breakout_volume_v1
# Strategy: 12h Camarilla Pivot Breakout with volume confirmation and 1d/1w trend filter
# Timeframe: 12h
# Leverage: 1.0
# Hypothesis: Camarilla pivot levels act as strong support/resistance. A breakout above R4 or below S4 with volume confirmation indicates momentum. Uses 1d EMA50 for short-term trend and 1w EMA200 for long-term regime filter to avoid counter-trend trades. Low frequency (~15-30/year) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_1w_camarilla_breakout_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d and 1w data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1d) < 50 or len(df_1w) < 50:
        return np.zeros(n)
    
    # 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 1w EMA(200) for regime filter
    close_1w = df_1w['close'].values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Calculate Camarilla pivot levels for each 12h bar using prior 12h bar's HLC
    # We need to shift the 12h data by one bar to use the previous bar's values for calculation
    high_12h_raw = get_htf_data(prices, '12h')['high'].values
    low_12h_raw = get_htf_data(prices, '12h')['low'].values
    close_12h_raw = get_htf_data(prices, '12h')['close'].values
    
    # Calculate pivot using previous bar's data (to avoid look-ahead)
    # For bar i, we use bar i-1's HLC
    pivot = (high_12h_raw[1:] + low_12h_raw[1:] + close_12h_raw[1:]) / 3
    range_val = high_12h_raw[1:] - low_12h_raw[1:]
    
    # Camarilla levels
    r4 = close_12h_raw[1:] + range_val * 1.5/2
    s4 = close_12h_raw[1:] - range_val * 1.5/2
    
    # Align to 12h timeframe (shifted by one bar to match calculation)
    # Since we used bar i-1 to calculate levels for bar i, we need to shift forward by 1
    r4_full = np.full(len(close_12h_raw), np.nan)
    s4_full = np.full(len(close_12h_raw), np.nan)
    r4_full[1:] = r4
    s4_full[1:] = s4
    
    # Now align to 12h timeframe
    r4_aligned = align_htf_to_ltf(prices, get_htf_data(prices, '12h'), r4_full)
    s4_aligned = align_htf_to_ltf(prices, get_htf_data(prices, '12h'), s4_full)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_avg_20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(ema_200_1w_aligned[i]) or 
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend filter: price above/below 1d EMA50
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Regime filter: price above/below 1w EMA200
        bull_regime = close[i] > ema_200_1w_aligned[i]
        bear_regime = close[i] < ema_200_1w_aligned[i]
        
        # Entry logic: Camarilla breakout + volume + trend/regime alignment
        if (close[i] > r4_aligned[i] and vol_confirm[i] and uptrend and bull_regime and position != 1):
            position = 1
            signals[i] = 0.25
        elif (close[i] < s4_aligned[i] and vol_confirm[i] and downtrend and bear_regime and position != -1):
            position = -1
            signals[i] = -0.25
        # Exit: price returns to pivot or regime/trend change
        elif position == 1 and (close[i] < (r4_aligned[i] + s4_aligned[i])/2 or not bull_regime or not uptrend):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > (r4_aligned[i] + s4_aligned[i])/2 or not bear_regime or not downtrend):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals