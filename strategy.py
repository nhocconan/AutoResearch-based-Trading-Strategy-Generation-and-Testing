#!/usr/bin/env python3
"""
12h_camarilla_pivot_1w_trend_volume_v1
Hypothesis: On 12h timeframe, use weekly Camarilla pivot levels (R3/S3 for mean reversion, R4/S4 for breakout) with EMA trend filter and volume confirmation. Enter long when price crosses above R3 with EMA20 > EMA50 and volume > 1.5x average; enter short when price crosses below S3 with EMA20 < EMA50 and volume > 1.5x average. Exit when price reaches opposite R4/S4 level or EMA crossover reverses. Uses weekly pivot structure to capture multi-day moves, with EMA filter to avoid counter-trend trades. Targets 15-30 trades/year to minimize fee drift.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_1w_trend_volume_v1"
timezone = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate EMA20 and EMA50 for trend filter
    ema20 = pd.Series(close).ewm(span=20, min_periods=20, adjust=False).mean().values
    ema50 = pd.Series(close).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Calculate weekly Camarilla pivot levels from prior week
    # Using weekly high, low, close from 1w timeframe
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    ph = df_1w['high'].values  # previous week high
    pl = df_1w['low'].values   # previous week low
    pc = df_1w['close'].values # previous week close
    
    # Calculate Camarilla levels for each week
    camarilla_h4 = pc + 1.1 * (ph - pl) / 2
    camarilla_l4 = pc - 1.1 * (ph - pl) / 2
    camarilla_h3 = pc + 1.1 * (ph - pl) / 4
    camarilla_l3 = pc - 1.1 * (ph - pl) / 4
    
    # Align to 12h timeframe (shifted by 1 week for look-ahead prevention)
    h4_12h = align_htf_to_ltf(prices, df_1w, camarilla_h4)
    l4_12h = align_htf_to_ltf(prices, df_1w, camarilla_l4)
    h3_12h = align_htf_to_ltf(prices, df_1w, camarilla_h3)
    l3_12h = align_htf_to_ltf(prices, df_1w, camarilla_l3)
    
    # Volume confirmation (24-period average on 12h = 12 days)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if required data not available
        if (np.isnan(ema20[i]) or np.isnan(ema50[i]) or 
            np.isnan(h3_12h[i]) or np.isnan(l3_12h[i]) or
            np.isnan(h4_12h[i]) or np.isnan(l4_12h[i]) or
            np.isnan(vol_ma[i]) or vol_ma[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 24-period average
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit conditions
            exit_long = False
            # Exit if price reaches L4 level (opposite extreme)
            if close[i] <= l4_12h[i]:
                exit_long = True
            # Exit if EMA20 crosses below EMA50 (trend reversal)
            elif ema20[i] < ema50[i] and ema20[i-1] >= ema50[i-1]:
                exit_long = True
            
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions
            exit_short = False
            # Exit if price reaches H4 level (opposite extreme)
            if close[i] >= h4_12h[i]:
                exit_short = True
            # Exit if EMA20 crosses above EMA50 (trend reversal)
            elif ema20[i] > ema50[i] and ema20[i-1] <= ema50[i-1]:
                exit_short = True
            
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price crosses above H3 with EMA20 > EMA50 and volume confirmation
            long_entry = False
            if (close[i] > h3_12h[i] and close[i-1] <= h3_12h[i-1] and
                ema20[i] > ema50[i] and vol_confirm):
                long_entry = True
            
            # Short entry: price crosses below L3 with EMA20 < EMA50 and volume confirmation
            short_entry = False
            if (close[i] < l3_12h[i] and close[i-1] >= l3_12h[i-1] and
                ema20[i] < ema50[i] and vol_confirm):
                short_entry = True
            
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
    
    return signals