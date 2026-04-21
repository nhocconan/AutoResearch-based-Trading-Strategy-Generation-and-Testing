#!/usr/bin/env python3
"""
12h_Camarilla_R1S1_Breakout_Volume_Regime_V1
Hypothesis: Use 12h Camarilla pivot levels (R1, S1) from 1d HTF for breakout entries, 
with 1d volume confirmation and 1d choppiness regime filter (CHOP > 61.8 = range). 
In ranging markets, trade mean reversion off Camarilla H3/L3 levels. 
ATR-based stoploss (2.0x ATR) manages risk. Position size 0.25 balances risk/return.
Designed to work in both bull (breakouts) and bear (mean reversion in range) markets.
Target 12-30 trades/year per symbol to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')  # for 1d Camarilla, volume, chop
    
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # === 1d Indicators ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Camarilla pivot levels (based on previous day)
    # R1 = close + 1.1*(high-low)/12
    # S1 = close - 1.1*(high-low)/12
    # H3 = close + 1.1*(high-low)/6
    # L3 = close - 1.1*(high-low)/6
    rng = high_1d - low_1d
    camarilla_r1 = close_1d + 1.1 * rng / 12
    camarilla_s1 = close_1d - 1.1 * rng / 12
    camarilla_h3 = close_1d + 1.1 * rng / 6
    camarilla_l3 = close_1d - 1.1 * rng / 6
    
    # Align Camarilla levels to 12h timeframe (completed 1d bar only)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # 1d volume MA (20-period) for spike confirmation
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # 1d Choppiness Index (CHOP) regime filter
    # CHOP = 100 * log10(sum(ATR(14)) / log10(highest-high-lowest-low)) / log10(N)
    # CHOP > 61.8 = ranging, CHOP < 38.2 = trending
    atr_1d_list = []
    for i in range(len(df_1d)):
        if i < 14:
            atr_1d_list.append(np.nan)
        else:
            tr = max(
                high_1d[i] - low_1d[i],
                abs(high_1d[i] - close_1d[i-1]),
                abs(low_1d[i] - close_1d[i-1])
            )
            atr_1d_list.append(tr)
    atr_1d = np.array(atr_1d_list)
    atr_sum = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    hh_ll = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values - \
            pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop_1d = 100 * np.log10(atr_sum / np.maximum(hh_ll, 1e-10)) / np.log10(14)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Align ATR for stoploss (1d ATR)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # === 12h Price Data ===
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_high_since_entry = 0.0
    lowest_low_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) 
            or np.isnan(chop_1d_aligned[i]) or np.isnan(atr_1d_aligned[i]) 
            or np.isnan(vol_ma_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        # Volume confirmation: current 12h volume > 1.5x 1d volume MA
        # Approximate 12h volume as 1d volume / 2 (since 2x12h = 1d)
        vol_approx = volume_1d[i//2] if i//2 < len(volume_1d) else volume_1d[-1]
        vol_ok = prices['volume'].iloc[i] > 1.5 * (vol_approx / 2)
        
        if position == 0:
            # Determine regime: CHOP > 61.8 = ranging (mean revert), else trending (breakout)
            if chop_1d_aligned[i] > 61.8:
                # Ranging market: mean reversion off H3/L3
                if price <= camarilla_l3_aligned[i] and vol_ok:
                    signals[i] = 0.25
                    position = 1
                    highest_high_since_entry = price
                elif price >= camarilla_h3_aligned[i] and vol_ok:
                    signals[i] = -0.25
                    position = -1
                    lowest_low_since_entry = price
            else:
                # Trending market: breakout of R1/S1
                if price > camarilla_r1_aligned[i-1] and vol_ok:
                    signals[i] = 0.25
                    position = 1
                    highest_high_since_entry = price
                elif price < camarilla_s1_aligned[i-1] and vol_ok:
                    signals[i] = -0.25
                    position = -1
                    lowest_low_since_entry = price
        
        elif position == 1:
            # Update highest high since entry
            if price > highest_high_since_entry:
                highest_high_since_entry = price
            # ATR trailing stop: exit if price drops 2.0*ATR from highest high since entry
            if price < highest_high_since_entry - 2.0 * atr_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Update lowest low since entry
            if price < lowest_low_since_entry:
                lowest_low_since_entry = price
            # ATR trailing stop: exit if price rises 2.0*ATR from lowest low since entry
            if price > lowest_low_since_entry + 2.0 * atr_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1S1_Breakout_Volume_Regime_V1"
timeframe = "12h"
leverage = 1.0