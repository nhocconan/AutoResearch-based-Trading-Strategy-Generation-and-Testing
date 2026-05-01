#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + Elder Ray combination with 1d trend filter
# Williams Alligator (jaw=13, teeth=8, lips=5 SMAs) identifies trend absence when lines are intertwined
# Elder Ray (Bull Power = High - EMA13, Bear Power = EMA13 - Low) measures trend strength
# In ranging markets (Alligator sleeping): fade extremes using Elder Ray divergences
# In trending markets (Alligator awakened): follow Elder Ray power direction
# 1d EMA34 filter ensures we only trade in alignment with higher timeframe trend
# Target: 50-150 total trades over 4 years (12-37/year) with balanced BTC/ETH performance

name = "6h_WilliamsAlligator_ElderRay_1dEMA34_TrendFilter_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 1d HTF data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA(34) for trend filter
    ema_1d_34 = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_34_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_34)
    
    # Williams Alligator: SMAs of median price (hlc3)
    hlc3 = (high + low + close) / 3.0
    
    # Jaw: 13-period SMA, 8-bar offset
    jaw = pd.Series(hlc3).rolling(window=13, min_periods=13).mean().shift(8).values
    # Teeth: 8-period SMA, 5-bar offset
    teeth = pd.Series(hlc3).rolling(window=8, min_periods=8).mean().shift(5).values
    # Lips: 5-period SMA, 3-bar offset
    lips = pd.Series(hlc3).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Align Alligator lines to 6h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = EMA13 - Low
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = ema13 - low
    
    # Alligator sleeping condition: lines are close together (intertwined)
    # Using standard deviation of the three lines as proxy for entanglement
    alligator_lines = np.column_stack([jaw_aligned, teeth_aligned, lips_aligned])
    alligator_std = np.nanstd(alligator_lines, axis=1)
    # Normalize by price level to make it adaptive
    alligator_sleeping = alligator_std < (close * 0.005)  # 0.5% of price
    
    # Alligator awakened condition: lines are separated and ordered
    # Bullish alignment: Lips > Teeth > Jaw
    # Bearish alignment: Jaw > Teeth > Lips
    bullish_alignment = (lips_aligned > teeth_aligned) & (teeth_aligned > jaw_aligned)
    bearish_alignment = (jaw_aligned > teeth_aligned) & (teeth_aligned > lips_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = 50  # Need enough for Alligator calculation
    
    for i in range(start_idx, n):
        if (np.isnan(ema_1d_34_aligned[i]) or np.isnan(jaw_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i])):
            signals[i] = 0.0
            continue
        
        # 1d trend filter: only trade long in uptrend, short in downtrend
        uptrend = close[i] > ema_1d_34_aligned[i]
        downtrend = close[i] < ema_1d_34_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            if alligator_sleeping[i]:
                # In ranging market: fade extremes using Elder Ray divergences
                # Long when bull power is rising but price makes lower low
                # Short when bear power is rising but price makes higher high
                if (bull_power[i] > bull_power[i-1]) and (low[i] < low[i-1]) and uptrend:
                    signals[i] = 0.25
                    position = 1
                elif (bear_power[i] > bear_power[i-1]) and (high[i] > high[i-1]) and downtrend:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            elif bullish_alignment[i] and uptrend:
                # In uptrend and Alligator bullish aligned: long on bull power expansion
                if bull_power[i] > bull_power[i-1]:
                    signals[i] = 0.25
                    position = 1
                else:
                    signals[i] = 0.0
            elif bearish_alignment[i] and downtrend:
                # In downtrend and Alligator bearish aligned: short on bear power expansion
                if bear_power[i] > bear_power[i-1]:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit conditions:
            # 1. Alligator sleeping and bear power expanding (range developing)
            # 2. Bearish alignment in uptrend (trend weakness)
            # 3. 1d trend turns down
            if (alligator_sleeping[i] and (bear_power[i] > bear_power[i-1])) or \
               (bearish_alignment[i] and uptrend) or \
               (not uptrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit conditions:
            # 1. Alligator sleeping and bull power expanding (range developing)
            # 2. Bullish alignment in downtrend (trend weakness)
            # 3. 1d trend turns up
            if (alligator_sleeping[i] and (bull_power[i] > bull_power[i-1])) or \
               (bullish_alignment[i] and downtrend) or \
               (not downtrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals