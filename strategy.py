#!/usr/bin/env python3
# 4h_1d_camarilla_breakout_v1
# Strategy: Camarilla pivot breakout with volume confirmation and 1-day trend filter
# Timeframe: 4h
# Leverage: 1.0
# Hypothesis: Price breakouts above/below Camarilla pivot levels (R4/S4) derived from
# daily high/low/close capture institutional momentum. Volume confirmation (volume > 
# 1.5x 20-period average) filters false breakouts. The 1-day EMA(50) trend filter ensures
# trades align with higher timeframe direction. Works in bull by capturing continuation
# breakouts and in bear by capturing breakdowns with volume confirmation.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_camarilla_breakout_v1"
timeframe = "4h"
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
    
    # Load daily data ONCE before loop for Camarilla levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Daily high, low, close for Camarilla calculation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels: R4 = close + 1.5*(high-low), S4 = close - 1.5*(high-low)
    camarilla_r4 = close_1d + 1.5 * (high_1d - low_1d)
    camarilla_s4 = close_1d - 1.5 * (high_1d - low_1d)
    
    # Align Camarilla levels to 4h timeframe (wait for daily close)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Daily EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: current volume > 1.5x 20-period average volume
    vol_series = pd.Series(volume)
    vol_avg_20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(1, n):  # Start from 1 to access previous close for breakout
        # Skip if any required data is invalid
        if (np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_confirm[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Breakout conditions: close breaks Camarilla levels
        bull_breakout = close[i] > camarilla_r4_aligned[i-1]  # Break above prior R4
        bear_breakout = close[i] < camarilla_s4_aligned[i-1]  # Break below prior S4
        
        # Trend filter: price above/below daily EMA50
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Entry logic: breakout + volume + trend alignment
        if bull_breakout and vol_confirm[i] and uptrend and position != 1:
            position = 1
            signals[i] = 0.25
        elif bear_breakout and vol_confirm[i] and downtrend and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: opposite breakout with volume confirmation
        elif position == 1 and bear_breakout and vol_confirm[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and bull_breakout and vol_confirm[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals