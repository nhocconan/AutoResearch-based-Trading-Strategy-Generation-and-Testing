#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) + weekly pivot bias.
# Elder Ray: Bull Power = High - EMA13, Bear Power = EMA13 - Low.
# Uses weekly pivot points (R1/S1, R2/S2) from higher timeframe to filter trades.
# In bullish weekly bias (price > weekly pivot): only take long when Bull Power > 0 and rising.
# In bearish weekly bias (price < weekly pivot): only take short when Bear Power > 0 and rising.
# Designed to capture momentum in direction of higher timeframe trend while avoiding counter-trend trades.
# Targets 12-37 trades/year (50-150 total over 4 years) with disciplined risk control.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data for pivot points (once before loop)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot points (standard formula)
    # P = (H + L + C)/3
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    r1_1w = 2 * pivot_1w - low_1w
    s1_1w = 2 * pivot_1w - high_1w
    r2_1w = pivot_1w + (high_1w - low_1w)
    s2_1w = pivot_1w - (high_1w - low_1w)
    
    # Align weekly pivot data to 6h timeframe
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    r2_1w_aligned = align_htf_to_ltf(prices, df_1w, r2_1w)
    s2_1w_aligned = align_htf_to_ltf(prices, df_1w, s2_1w)
    
    # Calculate EMA13 on 6h data for Elder Ray
    close = prices['close'].values
    ema13 = pd.Series(close).ewm(span=13, min_periods=13, adjust=False).mean().values
    
    # Calculate Elder Ray components
    high = prices['high'].values
    low = prices['low'].values
    bull_power = high - ema13  # Higher = stronger bullish momentum
    bear_power = ema13 - low   # Higher = stronger bearish momentum
    
    # Calculate rising momentum (current > previous)
    bull_power_rising = bull_power > np.roll(bull_power, 1)
    bear_power_rising = bear_power > np.roll(bear_power, 1)
    bull_power_rising[0] = False
    bear_power_rising[0] = False
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(13, n):
        # Skip if weekly pivot data not ready
        if (np.isnan(pivot_1w_aligned[i]) or 
            np.isnan(r1_1w_aligned[i]) or 
            np.isnan(s1_1w_aligned[i]) or
            np.isnan(ema13[i]) or
            np.isnan(bull_power[i]) or
            np.isnan(bear_power[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        pivot = pivot_1w_aligned[i]
        r1 = r1_1w_aligned[i]
        s1 = s1_1w_aligned[i]
        bull = bull_power[i]
        bear = bear_power[i]
        bull_rising = bull_power_rising[i]
        bear_rising = bear_power_rising[i]
        
        # Determine weekly bias
        is_bullish_bias = price > pivot  # Above weekly pivot = bullish bias
        is_bearish_bias = price < pivot  # Below weekly pivot = bearish bias
        
        if position == 0:
            # Enter long only in bullish weekly bias with rising bull power
            if is_bullish_bias and bull > 0 and bull_rising:
                signals[i] = 0.25
                position = 1
            # Enter short only in bearish weekly bias with rising bear power
            elif is_bearish_bias and bear > 0 and bear_rising:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when bull power fails or turns bearish
                if bull <= 0 or not bull_rising:
                    exit_signal = True
                # Also exit if price hits weekly S1 (strong support)
                elif price <= s1:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when bear power fails or turns bullish
                if bear <= 0 or not bear_rising:
                    exit_signal = True
                # Also exit if price hits weekly R1 (strong resistance)
                elif price >= r1:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_ElderRay_WeeklyPivot_Bias"
timeframe = "6h"
leverage = 1.0