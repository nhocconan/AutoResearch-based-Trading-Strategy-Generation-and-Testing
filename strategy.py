#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout with 4h trend filter and session filter (08-20 UTC)
# Uses 1d Camarilla pivot levels (H3/L3) for breakout triggers, confirmed by 4h trend direction
# Only trades during active session (08-20 UTC) to reduce noise and false breakouts
# Position size 0.20 to manage drawdown and enable multiple concurrent positions
# Target: 60-150 total trades over 4 years (15-37/year) to balance edge and fee drag
# Works in both bull/bear: 4h trend filter ensures we trade with higher timeframe momentum

name = "1h_1d_camarilla_4htrend_session_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Precompute session filter (08-20 UTC)
    hours = open_time.dt.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 1d data ONCE before loop for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels (H3, L3) from 1d OHLC (using previous day's data)
    camarilla_h3 = np.full(len(df_1d), np.nan)
    camarilla_l3 = np.full(len(df_1d), np.nan)
    
    for i in range(1, len(df_1d)):
        prev_high = df_1d['high'].iloc[i-1]
        prev_low = df_1d['low'].iloc[i-1]
        prev_close = df_1d['close'].iloc[i-1]
        range_val = prev_high - prev_low
        camarilla_h3[i] = prev_close + range_val * 1.1 / 4
        camarilla_l3[i] = prev_close - range_val * 1.1 / 4
    
    # Align 1d Camarilla levels to 1h timeframe
    camarilla_h3_1h = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_1h = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Load 4h data ONCE before loop for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h EMA(20) for trend filter
    close_4h = df_4h['close'].values
    ema_4h = np.full(len(df_4h), np.nan)
    if len(df_4h) >= 20:
        ema_4h[19] = np.mean(close_4h[0:20])
        for i in range(20, len(df_4h)):
            ema_4h[i] = (close_4h[i] * 2) / (20 + 1) + ema_4h[i-1] * (19) / (20 + 1)
    
    # Align 4h EMA to 1h timeframe
    ema_4h_1h = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):  # Start after warmup
        # Skip if any required data is invalid or outside session
        if (np.isnan(camarilla_h3_1h[i]) or 
            np.isnan(camarilla_l3_1h[i]) or 
            np.isnan(ema_4h_1h[i]) or 
            not in_session.iloc[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below Camarilla L3 OR 4h trend turns bearish
            if close[i] < camarilla_l3_1h[i] or close[i] < ema_4h_1h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: price closes above Camarilla H3 OR 4h trend turns bullish
            if close[i] > camarilla_h3_1h[i] or close[i] > ema_4h_1h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Entry: Camarilla breakout with 4h trend filter
            # Long: price above H3 AND above 4h EMA (bullish alignment)
            if close[i] > camarilla_h3_1h[i] and close[i] > ema_4h_1h[i]:
                position = 1
                signals[i] = 0.20
            # Short: price below L3 AND below 4h EMA (bearish alignment)
            elif close[i] < camarilla_l3_1h[i] and close[i] < ema_4h_1h[i]:
                position = -1
                signals[i] = -0.20
    
    return signals