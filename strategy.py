#!/usr/bin/env python3
"""
1d_WilliamsAlligator_ElderRay_WeeklyTrend_Regime_v1
Hypothesis: Combine Williams Alligator (trend direction) with Elder Ray Index (bull/bear power) on 1d, filtered by weekly trend and choppiness regime. Designed for low trade frequency (~10-25/year) to minimize fee drag while capturing sustained moves in both bull and bear markets. Uses 1d primary timeframe with 1w HTF for regime and trend context.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # === Williams Alligator on 1d (Jaw=13, Teeth=8, Lips=5, all smoothed) ===
    df_1d = get_htf_data(prices, '1d')  # Need 1d data for Alligator calculation
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate median price (typical price) for Alligator
    typical_price = (df_1d['high'].values + df_1d['low'].values + df_1d['close'].values) / 3.0
    
    # Alligator lines: Jaw (13,8), Teeth (8,5), Lips (5,3) - all smoothed
    jaw = pd.Series(typical_price).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth = pd.Series(typical_price).rolling(window=8, min_periods=8).mean().shift(5).values
    lips = pd.Series(typical_price).rolling(window=5, min_periods=5).mean().shift(3).values
    
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # === Elder Ray Index on 1d (Bull Power = High - EMA13, Bear Power = Low - EMA13) ===
    close_1d = df_1d['close'].values
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema_13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_13_1d)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    bull_power = high_1d - ema_13_1d
    bear_power = low_1d - ema_13_1d
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # === Weekly trend: 34-period EMA on weekly close ===
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # === Choppiness regime filter on 1d (CHOP > 61.8 = range, CHOP < 38.2 = trend) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Highest high and lowest low over 14 periods
    hh_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Chopiness Index: CHOP = 100 * log10(sum(ATR14) / (HH14 - LL14)) / log10(14)
    sum_atr_14 = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    range_14 = hh_14 - ll_14
    # Avoid division by zero
    range_14[range_14 == 0] = 1e-10
    chop = 100 * np.log10(sum_atr_14 / range_14) / np.log10(14)
    chop[np.isnan(chop)] = 50.0  # neutral when undefined
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(chop_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        bull_power_val = bull_power_aligned[i]
        bear_power_val = bear_power_aligned[i]
        weekly_ema = ema_34_1w_aligned[i]
        chop_val = chop_aligned[i]
        
        # Alligator alignment: Lips > Teeth > Jaw = uptrend, Lips < Teeth < Jaw = downtrend
        alligator_long = lips_val > teeth_val > jaw_val
        alligator_short = lips_val < teeth_val < jaw_val
        
        # Elder Ray: Bull Power > 0 and Bear Power < 0 confirms trend strength
        elder_long = bull_power_val > 0 and bear_power_val < 0
        elder_short = bull_power_val < 0 and bear_power_val > 0
        
        # Weekly trend filter: price above/below weekly EMA34
        weekly_long = price_close > weekly_ema
        weekly_short = price_close < weekly_ema
        
        # Regime filter: only trade in trending markets (CHOP < 38.2) or strong momentum in range
        trending_regime = chop_val < 38.2
        strong_momentum = abs(bull_power_val) > 0.5 * price_close or abs(bear_power_val) > 0.5 * price_close
        
        if position == 0:
            # Long: Alligator uptrend + Elder Ray bullish + weekly trend up + (trending OR strong momentum)
            if (alligator_long and elder_long and weekly_long and 
                (trending_regime or strong_momentum)):
                signals[i] = 0.25
                position = 1
                entry_price = price_close
            # Short: Alligator downtrend + Elder Ray bearish + weekly trend down + (trending OR strong momentum)
            elif (alligator_short and elder_short and weekly_short and 
                  (trending_regime or strong_momentum)):
                signals[i] = -0.25
                position = -1
                entry_price = price_close
        
        elif position != 0:
            # Exit conditions: Alligator reversal OR Elder Ray divergence OR weekly trend failure
            exit_long = (not alligator_long) or (bull_power_val <= 0) or (price_close < weekly_ema)
            exit_short = (not alligator_short) or (bear_power_val >= 0) or (price_close > weekly_ema)
            
            if position == 1 and exit_long:
                signals[i] = 0.0
                position = 0
            elif position == -1 and exit_short:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_WilliamsAlligator_ElderRay_WeeklyTrend_Regime_v1"
timeframe = "1d"
leverage = 1.0