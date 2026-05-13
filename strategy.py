#!/usr/bin/env python3
# 6h_ElderRay_1dTrend_Filter
# Hypothesis: Elder Ray (Bull/Bear Power) combined with 1d trend filter to capture institutional
# buying/selling pressure. Goes long when Bear Power shows weakening bears + 1d uptrend,
# short when Bull Power shows weakening bulls + 1d downtrend. Volume spike confirms.
# Works in bull (buy weakening bear pressure) and bear (sell weakening bull pressure) markets.
# Low frequency due to Elder Ray smoothing and volume confirmation requirement.

name = "6h_ElderRay_1dTrend_Filter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get 1d data for trend filter and EMA13 for Elder Ray
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA13 for Elder Ray calculation
    ema13_1d = pd.Series(df_1d['close']).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # 1d trend filter: EMA50
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d indicators to 6h timeframe
    ema13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema13_1d)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate Elder Ray components on 1d
    # Bull Power = High - EMA13
    bull_power = df_1d['high'].values - ema13_1d
    # Bear Power = Low - EMA13
    bear_power = df_1d['low'].values - ema13_1d
    
    # Align Elder Ray to 6h
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Volume spike: volume > 1.8 * 20-period average (moderate threshold)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 1.8 * vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required value is NaN
        if (np.isnan(ema13_1d_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(bull_power_aligned[i]) or 
            np.isnan(bear_power_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Trend conditions
        uptrend = close[i] > ema50_1d_aligned[i]
        downtrend = close[i] < ema50_1d_aligned[i]
        
        # Elder Ray conditions
        # Bull Power declining but still positive = weakening bulls
        bull_power_declining = (i >= 1 and bull_power_aligned[i] < bull_power_aligned[i-1] and bull_power_aligned[i] > 0)
        # Bear Power rising but still negative = weakening bears  
        bear_power_rising = (i >= 1 and bear_power_aligned[i] > bear_power_aligned[i-1] and bear_power_aligned[i] < 0)
        
        # Volume confirmation
        vol_spike = volume_spike[i]

        if position == 0:
            # LONG: Bears weakening (Bear Power rising but negative) + uptrend + volume spike
            if bear_power_rising and uptrend and vol_spike:
                signals[i] = 0.25
                position = 1
            # SHORT: Bulls weakening (Bull Power declining but positive) + downtrend + volume spike
            elif bull_power_declining and downtrend and vol_spike:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Bulls strengthening OR trend reversal
            if bull_power_declining or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Bears strengthening OR trend reversal
            if bear_power_rising or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals