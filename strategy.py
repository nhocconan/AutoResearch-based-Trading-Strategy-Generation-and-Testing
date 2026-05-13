#!/usr/bin/env python3
# Hypothesis: 12h Williams Alligator with 1d EMA34 trend filter and volume spike confirmation.
# Long when Alligator jaws (13-period SMMA) < teeth (8-period SMMA) < lips (5-period SMMA) AND close > 1d EMA34 AND volume > 2.0x average
# Short when Alligator jaws > teeth > lips AND close < 1d EMA34 AND volume > 2.0x average
# Exit when Alligator lines cross (jaws-teeth or teeth-lips) OR trend reversal (price crosses 1d EMA34)
# Uses 12h timeframe (target: 50-150 total trades over 4 years = 12-37/year) with daily trend filter for BTC/ETH resilience.
# Williams Alligator identifies trending markets via SMMA alignment; EMA34 filters higher-timeframe trend; volume spike confirms breakout authenticity.

name = "12h_WilliamsAlligator_1dEMA34_Volume_v1"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(source, length):
    """Smoothed Moving Average (SMMA) as used in Williams Alligator"""
    if length < 1:
        return np.full_like(source, np.nan)
    result = np.full_like(source, np.nan)
    if len(source) < length:
        return result
    # First value is simple SMA
    result[length-1] = np.nansum(source[:length]) / length
    # Subsequent values: SMMA = (PREV_SMMA * (LENGTH-1) + CLOSE) / LENGTH
    for i in range(length, len(source)):
        if np.isnan(result[i-1]):
            result[i] = np.nansum(source[i-length+1:i+1]) / length
        else:
            result[i] = (result[i-1] * (length-1) + source[i]) / length
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Williams Alligator calculation (primary timeframe)
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate Williams Alligator on 12h close: jaws (13), teeth (8), lips (5) SMMA
    lips = smma(close_12h, 5)
    teeth = smma(close_12h, 8)
    jaws = smma(close_12h, 13)
    
    # Align Alligator lines to 12h timeframe (already aligned since calculated on 12h)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth)
    jaws_aligned = align_htf_to_ltf(prices, df_12h, jaws)
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA(34) on 1d close for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume filter: current 12h volume > 2.0x 20-period average (spike confirmation)
    vol_ma_12h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (2.0 * vol_ma_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after sufficient data for EMA and Alligator
        # Skip if any required data is NaN
        if (np.isnan(jaws_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma_12h[i])):
            signals[i] = 0.0
            continue
        
        # Alligator conditions: jaws < teeth < lips (bullish alignment) OR jaws > teeth > lips (bearish alignment)
        bullish_alignment = jaws_aligned[i] < teeth_aligned[i] and teeth_aligned[i] < lips_aligned[i]
        bearish_alignment = jaws_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > lips_aligned[i]
        
        if position == 0:
            # LONG: bullish Alligator alignment AND close > 1d EMA34 AND volume spike
            if bullish_alignment and close[i] > ema34_1d_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: bearish Alligator alignment AND close < 1d EMA34 AND volume spike
            elif bearish_alignment and close[i] < ema34_1d_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Alligator lines cross (jaws-teeth or teeth-lips) OR trend reversal (close < 1d EMA34)
            if (jaws_aligned[i] >= teeth_aligned[i] or teeth_aligned[i] >= lips_aligned[i]) or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Alligator lines cross (jaws-teeth or teeth-lips) OR trend reversal (close > 1d EMA34)
            if (jaws_aligned[i] <= teeth_aligned[i] or teeth_aligned[i] <= lips_aligned[i]) or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals