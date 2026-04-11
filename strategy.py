#!/usr/bin/env python3
# 12h_1w_camarilla_pivot_volume_v1
# Strategy: 12-hour Camarilla pivot breakout with 1-week trend filter
# Timeframe: 12h
# Leverage: 1.0
# Hypothesis: Price breakouts above/below weekly Camarilla pivot levels (R4/S4) with volume > 2x average
# capture institutional momentum. The 1-week EMA(50) trend filter ensures trades align with higher timeframe
# direction, reducing false breakouts. Works in bull by catching continuation breakouts and in bear by
# capturing breakdowns with volume confirmation. Targets 12-37 trades/year to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w_camarilla_pivot_volume_v1"
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
    
    # Load weekly data ONCE before loop for Camarilla pivots and trend filter
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly OHLC for Camarilla calculation
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly Camarilla levels: R4 = C + (H-L)*1.1/2, S4 = C - (H-L)*1.1/2
    camarilla_r4 = close_1w + (high_1w - low_1w) * 1.1 / 2
    camarilla_s4 = close_1w - (high_1w - low_1w) * 1.1 / 2
    
    # Align weekly Camarilla levels to 12h timeframe (wait for weekly close)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s4)
    
    # Weekly EMA(50) for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # 12h Relative Volume (RVOL): current volume / 20-period average volume
    vol_series = pd.Series(volume)
    vol_avg_20 = vol_series.rolling(window=20, min_periods=20).mean().values
    rvol = volume / (vol_avg_20 + 1e-10)  # Avoid division by zero
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after RVOL warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(rvol[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Breakout conditions
        bull_breakout = close[i] > camarilla_r4_aligned[i-1]  # Break above prior R4
        bear_breakout = close[i] < camarilla_s4_aligned[i-1]   # Break below prior S4
        
        # Volume confirmation: RVOL > 2.0
        vol_confirm = rvol[i] > 2.0
        
        # Trend filter: price above/below weekly EMA50
        uptrend = close[i] > ema_50_1w_aligned[i]
        downtrend = close[i] < ema_50_1w_aligned[i]
        
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