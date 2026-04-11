#!/usr/bin/env python3
"""
12h_1d_williams_alligator_v1
Strategy: Williams Alligator on 12h with 1d trend filter and volume confirmation
Timeframe: 12h
Leverage: 1.0
Hypothesis: Uses Williams Alligator (Jaw/Teeth/Lips) on 12h to detect trend changes. Enters long when Lips > Teeth > Jaw (bullish alignment) with 1d EMA50 uptrend and volume > 1.5x average. Short when Lips < Teeth < Jaw with 1d EMA50 downtrend. Exits when Alligator lines re-cross or trend weakens. Designed to catch sustained trends while avoiding whipsaws in ranging markets. Target: 50-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_williams_alligator_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load higher timeframe data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_12h) < 50 or len(df_1d) < 50:
        return np.zeros(n)
    
    # Williams Alligator on 12h: SMMA (Smoothed Moving Average)
    # Jaw: 13-period SMMA shifted 8 bars
    # Teeth: 8-period SMMA shifted 5 bars  
    # Lips: 5-period SMMA shifted 3 bars
    close_12h = df_12h['close'].values
    
    # Calculate SMMA (Smoothed Moving Average)
    def smma(values, period):
        sma = np.full_like(values, np.nan, dtype=np.float64)
        sma[period-1] = np.mean(values[:period])
        for i in range(period, len(values)):
            sma[i] = (sma[i-1] * (period-1) + values[i]) / period
        return sma
    
    smma_13 = smma(close_12h, 13)
    smma_8 = smma(close_12h, 8)
    smma_5 = smma(close_12h, 5)
    
    # Shift as per Alligator definition
    jaw = np.roll(smma_13, 8)   # Jaw: 13-period SMMA shifted 8 bars forward
    teeth = np.roll(smma_8, 5)  # Teeth: 8-period SMMA shifted 5 bars forward
    lips = np.roll(smma_5, 3)   # Lips: 5-period SMMA shifted 3 bars forward
    
    # Align to lower timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips)
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_surge = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        price_close = close[i]
        
        # Alligator alignment signals
        bullish_alignment = (lips_aligned[i] > teeth_aligned[i]) and (teeth_aligned[i] > jaw_aligned[i])
        bearish_alignment = (lips_aligned[i] < teeth_aligned[i]) and (teeth_aligned[i] < jaw_aligned[i])
        
        # Trend filters
        uptrend_1d = price_close > ema_50_1d_aligned[i]
        downtrend_1d = price_close < ema_50_1d_aligned[i]
        
        # Entry conditions
        long_signal = bullish_alignment and uptrend_1d and volume_surge[i]
        short_signal = bearish_alignment and downtrend_1d and volume_surge[i]
        
        # Exit when Alligator lines re-cross (trend weakening) or trend reverses
        exit_long = position == 1 and (not bullish_alignment or not uptrend_1d)
        exit_short = position == -1 and (not bearish_alignment or not downtrend_1d)
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
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
"""12h_1d_williams_alligator_v1
Strategy: Williams Alligator on 12h with 1d trend filter and volume confirmation
Timeframe: 12h
Leverage: 1.0
Hypothesis: Uses Williams Alligator (Jaw/Teeth/Lips) on 12h to detect trend changes. Enters long when Lips > Teeth > Jaw (bullish alignment) with 1d EMA50 uptrend and volume > 1.5x average. Short when Lips < Teeth < Jaw with 1d EMA50 downtrend. Exits when Alligator lines re-cross or trend weakens. Designed to catch sustained trends while avoiding whipsaws in ranging markets. Target: 50-150 total trades over 4 years.
"""