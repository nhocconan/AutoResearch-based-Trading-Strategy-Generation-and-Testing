#!/usr/bin/env python3
# 12h_1w_camarilla_breakout_v1
# Strategy: Camarilla pivot breakout on 12h with weekly trend filter and volume confirmation
# Timeframe: 12h
# Leverage: 1.0
# Hypothesis: Camarilla pivot levels act as strong support/resistance. Price breaking above/below these levels with volume and weekly trend alignment captures institutional moves. Weekly filter avoids counter-trend trades. Works in bull by catching breakouts in uptrend, and in bear by catching breakdowns in downtrend.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w_camarilla_breakout_v1"
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
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Weekly EMA(21) for trend filter
    close_1w = df_1w['close'].values
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    weekly_uptrend = ema_21_1w > np.roll(ema_21_1w, 1)  # Rising EMA
    weekly_uptrend[0] = False  # First value invalid
    weekly_uptrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_uptrend)
    
    # Daily data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla formulas
    range_1d = high_1d - low_1d
    camarilla_h4 = close_1d + range_1d * 1.1 / 2
    camarilla_l4 = close_1d - range_1d * 1.1 / 2
    camarilla_h3 = close_1d + range_1d * 1.1 / 4
    camarilla_l3 = close_1d - range_1d * 1.1 / 4
    
    # Align Camarilla levels to 12h timeframe
    camarilla_h4_12h = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_12h = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    camarilla_h3_12h = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_12h = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Volume average (24-period ~ 12 days) for confirmation
    vol_avg = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_spike = volume > (1.5 * vol_avg)  # Volume spike filter
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(24, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h4_12h[i]) or np.isnan(camarilla_l4_12h[i]) or 
            np.isnan(weekly_uptrend_aligned[i]) or np.isnan(vol_avg[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Current values
        price_now = close[i]
        volume_now = volume[i]
        
        # Camarilla levels at current bar
        h4 = camarilla_h4_12h[i]
        l4 = camarilla_l4_12h[i]
        h3 = camarilla_h3_12h[i]
        l3 = camarilla_l3_12h[i]
        
        # Breakout conditions with volume and trend
        bull_break = (price_now > h4) and volume_now and weekly_uptrend_aligned[i]
        bear_break = (price_now < l4) and volume_now and not weekly_uptrend_aligned[i]
        
        # Exit conditions: reverse signal or re-entry into Camarilla body
        exit_long = position == 1 and (price_now < h3 or bear_break)
        exit_short = position == -1 and (price_now > l3 or bull_break)
        
        # Trading logic
        if bull_break and position != 1:
            position = 1
            signals[i] = 0.25
        elif bear_break and position != -1:
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