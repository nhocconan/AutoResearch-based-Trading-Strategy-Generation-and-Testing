#!/usr/bin/env python3
"""
1d_weekly_pivot_breakout_volume_v1
Hypothesis: On 1d timeframe, use weekly pivot levels (R1/S1 for mean reversion, R2/S2 for breakout) with 50/200 EMA trend filter and volume confirmation. Enter long when price closes above R1 with EMA50 > EMA200 and volume > 1.5x average; enter short when price closes below S1 with EMA50 < EMA200 and volume > 1.5x average. Exit when price reaches opposite S2/R2 level or EMA crossover reverses. This strategy combines mean reversion at weekly extremes with breakout continuation, using volume to confirm institutional participation. Works in bull/bear via EMA trend filter and pivot level structure. Targets 10-25 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_weekly_pivot_breakout_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate EMA50 and EMA200 for trend filter
    ema50 = pd.Series(close).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema200 = pd.Series(close).ewm(span=200, min_periods=200, adjust=False).mean().values
    
    # Calculate weekly pivot levels from prior week
    # Using weekly high, low, close from 1w timeframe
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    ph = df_1w['high'].values  # previous week high
    pl = df_1w['low'].values   # previous week low
    pc = df_1w['close'].values # previous week close
    
    # Calculate weekly pivot levels
    pivot = (ph + pl + pc) / 3
    r1 = 2 * pivot - pl
    s1 = 2 * pivot - ph
    r2 = pivot + (ph - pl)
    s2 = pivot - (ph - pl)
    
    # Align to 1d timeframe (shifted by 1 week for look-ahead prevention)
    r1_1d = align_htf_to_ltf(prices, df_1w, r1)
    s1_1d = align_htf_to_ltf(prices, df_1w, s1)
    r2_1d = align_htf_to_ltf(prices, df_1w, r2)
    s2_1d = align_htf_to_ltf(prices, df_1w, s2)
    
    # Volume confirmation (20-period average on 1d = 20 days)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(200, n):
        # Skip if required data not available
        if (np.isnan(ema50[i]) or np.isnan(ema200[i]) or 
            np.isnan(r1_1d[i]) or np.isnan(s1_1d[i]) or
            np.isnan(r2_1d[i]) or np.isnan(s2_1d[i]) or
            np.isnan(vol_ma[i]) or vol_ma[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit conditions
            exit_long = False
            # Exit if price reaches S2 level (opposite extreme)
            if close[i] <= s2_1d[i]:
                exit_long = True
            # Exit if EMA50 crosses below EMA200 (trend reversal)
            elif ema50[i] < ema200[i] and ema50[i-1] >= ema200[i-1]:
                exit_long = True
            
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions
            exit_short = False
            # Exit if price reaches R2 level (opposite extreme)
            if close[i] >= r2_1d[i]:
                exit_short = True
            # Exit if EMA50 crosses above EMA200 (trend reversal)
            elif ema50[i] > ema200[i] and ema50[i-1] <= ema200[i-1]:
                exit_short = True
            
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price closes above R1 with EMA50 > EMA200 and volume confirmation
            long_entry = False
            if (close[i] > r1_1d[i] and close[i-1] <= r1_1d[i-1] and
                ema50[i] > ema200[i] and vol_confirm):
                long_entry = True
            
            # Short entry: price closes below S1 with EMA50 < EMA200 and volume confirmation
            short_entry = False
            if (close[i] < s1_1d[i] and close[i-1] >= s1_1d[i-1] and
                ema50[i] < ema200[i] and vol_confirm):
                short_entry = True
            
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
    
    return signals