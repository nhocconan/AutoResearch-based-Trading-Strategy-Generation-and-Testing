#!/usr/bin/env python3
# 12h_1d_camarilla_breakout_v1
# Strategy: 12-hour Camarilla pivot breakout with 1-day trend filter
# Timeframe: 12h
# Leverage: 1.0
# Hypothesis: Price breaking above/below daily Camarilla pivot levels (H4/L4) with
# volume confirmation (RVOL > 1.5) captures institutional moves. The 1-day EMA(50)
# trend filter ensures trades align with higher timeframe direction, reducing false
# breakouts in sideways markets. Works in bull by catching continuation breakouts
# and in bear by capturing breakdowns with volume confirmation.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_camarilla_breakout_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop for Camarilla pivot and trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Daily OHLC for Camarilla pivot calculation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily Camarilla levels: H4, L4
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # H4 = Close + Range * 1.1/2
    # L4 = Close - Range * 1.1/2
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    camarilla_h4 = close_1d + range_1d * 1.1 / 2.0
    camarilla_l4 = close_1d - range_1d * 1.1 / 2.0
    
    # Daily EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1D indicators to 12H timeframe
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 12H Relative Volume (RVOL): current volume / 20-period average volume
    vol_series = pd.Series(volume)
    vol_avg_20 = vol_series.rolling(window=20, min_periods=20).mean().values
    rvol = volume / (vol_avg_20 + 1e-10)  # Avoid division by zero
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after RVOL warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(rvol[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Breakout conditions
        bull_breakout = close[i] > camarilla_h4_aligned[i-1]  # Break above prior H4
        bear_breakout = close[i] < camarilla_l4_aligned[i-1]  # Break below prior L4
        
        # Volume confirmation: RVOL > 1.5
        vol_confirm = rvol[i] > 1.5
        
        # Trend filter: price above/below daily EMA50
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Entry logic: breakout + volume + trend alignment
        if bull_breakout and vol_confirm and uptrend and position != 1:
            position = 1
            signals[i] = 0.25
        elif bear_breakout and vol_confirm and downtrend and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: opposite breakout with volume confirmation
        elif position == 1 and bear_breakout and vol_confirm:
            position = 0
            signals[i] = 0.0
        elif position == -1 and bull_breakout and vol_confirm:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals