#!/usr/bin/env python3
"""
4h_camarilla_pivot_1d_trend_volume_v1
Hypothesis: On 4h timeframe, use daily Camarilla pivot levels (H3/L3 for mean reversion, H4/L4 for breakout) with EMA trend filter and volume confirmation. Enter long when price crosses above H3 with EMA20 > EMA50 and volume > 1.5x average; enter short when price crosses below L3 with EMA20 < EMA50 and volume > 1.5x average. Exit when price reaches opposite L4/H4 level or EMA crossover reverses. This strategy combines mean reversion at extreme daily levels with breakout continuation, using volume to confirm institutional participation. Works in bull/bear via EMA trend filter and pivot level structure. Targets 20-50 trades/year to minimize fee drift.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_camarilla_pivot_1d_trend_volume_v1"
timeframe = "4h"
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
    
    # Calculate daily Camarilla pivot levels from prior day
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    ph = df_1d['high'].values  # previous day high
    pl = df_1d['low'].values   # previous day low
    pc = df_1d['close'].values # previous day close
    
    # Calculate Camarilla levels for each day
    camarilla_h4 = pc + 1.1 * (ph - pl) / 2
    camarilla_l4 = pc - 1.1 * (ph - pl) / 2
    camarilla_h3 = pc + 1.1 * (ph - pl) / 4
    camarilla_l3 = pc - 1.1 * (ph - pl) / 4
    camarilla_h2 = pc + 1.1 * (ph - pl) / 6
    camarilla_l2 = pc - 1.1 * (ph - pl) / 6
    camarilla_h1 = pc + 1.1 * (ph - pl) / 12
    camarilla_l1 = pc - 1.1 * (ph - pl) / 12
    
    # Align to 4h timeframe (shifted by 1 day for look-ahead prevention)
    h4_4h = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    l4_4h = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    h3_4h = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_4h = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Volume confirmation (12-period average on 4h = 2 days)
    vol_ma = pd.Series(volume).rolling(window=12, min_periods=12).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if required data not available
        if (np.isnan(ema20[i]) or np.isnan(ema50[i]) or 
            np.isnan(h3_4h[i]) or np.isnan(l3_4h[i]) or
            np.isnan(h4_4h[i]) or np.isnan(l4_4h[i]) or
            np.isnan(vol_ma[i]) or vol_ma[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 12-period average
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit conditions
            exit_long = False
            # Exit if price reaches L4 level (opposite extreme)
            if close[i] <= l4_4h[i]:
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
            if close[i] >= h4_4h[i]:
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
            if (close[i] > h3_4h[i] and close[i-1] <= h3_4h[i-1] and
                ema20[i] > ema50[i] and vol_confirm):
                long_entry = True
            
            # Short entry: price crosses below L3 with EMA20 < EMA50 and volume confirmation
            short_entry = False
            if (close[i] < l3_4h[i] and close[i-1] >= l3_4h[i-1] and
                ema20[i] < ema50[i] and vol_confirm):
                short_entry = True
            
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
    
    return signals