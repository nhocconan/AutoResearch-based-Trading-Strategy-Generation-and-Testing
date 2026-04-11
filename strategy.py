#!/usr/bin/env python3
# 4h_1d_camarilla_breakout_v1
# Strategy: 4-hour Camarilla pivot breakout with daily trend filter and volume confirmation
# Timeframe: 4h
# Leverage: 1.0
# Hypothesis: Camarilla pivot levels (H3/L3, H4/L4) act as strong support/resistance.
# Breakouts above H3 or below L3 with volume confirmation and daily trend alignment
# capture institutional moves. Works in bull by catching breakouts in uptrend,
# and in bear by catching breakdowns in downtrend.

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
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Calculate Camarilla levels for previous day
    # H4 = close + 1.5 * (high - low)
    # L4 = close - 1.5 * (high - low)
    # H3 = close + 1.125 * (high - low)
    # L3 = close - 1.125 * (high - low)
    range_1d = prev_high - prev_low
    H4 = prev_close + 1.5 * range_1d
    L4 = prev_close - 1.5 * range_1d
    H3 = prev_close + 1.125 * range_1d
    L3 = prev_close - 1.125 * range_1d
    
    # Align daily Camarilla levels to 4h timeframe (no extra delay needed for pivot points)
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    
    # Daily EMA(50) for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume average (20-period) for confirmation
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_avg)  # Volume spike filter
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after EMA warmup
        # Skip if any required data is invalid
        if (np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_avg[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Current price
        price_now = close[i]
        
        # Breakout conditions
        bull_breakout = (price_now > H3_aligned[i]) and vol_spike[i]
        bear_breakout = (price_now < L3_aligned[i]) and vol_spike[i]
        
        # Trend filter: only trade in direction of daily EMA(50)
        bull_mode = close[i] > ema_50_aligned[i]
        bear_mode = close[i] < ema_50_aligned[i]
        
        # Exit conditions: price returns to camillia pivot levels
        exit_long = position == 1 and (price_now < H3_aligned[i] * 0.995)
        exit_short = position == -1 and (price_now > L3_aligned[i] * 1.005)
        
        # Trading logic
        if bull_breakout and bull_mode and position != 1:
            position = 1
            signals[i] = 0.25
        elif bear_breakout and bear_mode and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals