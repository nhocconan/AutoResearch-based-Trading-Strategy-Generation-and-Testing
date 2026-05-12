#!/usr/bin/env python3
"""
1D Williams Alligator with Volume Spike and 1W Trend Filter
Trades the jaw-teeth-lips configuration with volume confirmation and weekly trend alignment.
Works in both bull and bear markets: Alligator identifies trend direction, volume confirms breakout strength,
and weekly filter ensures alignment with higher timeframe momentum.
Target: 15-25 trades/year (60-100 total over 4 years).
"""
name = "1D_WilliamsAlligator_VolumeSpike_1WTrend"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1W DATA FOR TREND FILTER ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Weekly EMA50 for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # === WILLIAMS ALLIGATOR (13,8,5 SMMA) ===
    # Smoothed Moving Average (SMMA) calculation
    def smma(data, period):
        result = np.full_like(data, np.nan, dtype=float)
        if len(data) >= period:
            # First value is simple average
            result[period-1] = np.mean(data[:period])
            # Subsequent values: SMMA = (Prev SMMA * (period-1) + Current Price) / period
            for i in range(period, len(data)):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    # Alligator lines: Jaw (13), Teeth (8), Lips (5)
    jaw = smma(high, 13)  # Using high for jaw
    teeth = smma(low, 8)  # Using low for teeth
    lips = smma(close, 5)  # Using close for lips
    
    jaw_aligned = align_htf_to_ltf(prices, pd.DataFrame({'high': high}), jaw)
    teeth_aligned = align_htf_to_ltf(prices, pd.DataFrame({'low': low}), teeth)
    lips_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), lips)
    
    # === VOLUME CONFIRMATION (20-period) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 13, 8, 5, 20)  # Max of all periods
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(jaw_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Alligator aligned (Lips > Teeth > Jaw) AND price above all lines AND volume spike AND weekly uptrend
            if (lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i] and
                close[i] > lips_aligned[i] and
                close[i] > ema50_1w_aligned[i] and
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Alligator aligned (Lips < Teeth < Jaw) AND price below all lines AND volume spike AND weekly downtrend
            elif (lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i] and
                  close[i] < lips_aligned[i] and
                  close[i] < ema50_1w_aligned[i] and
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Alligator convergence (Lips < Teeth) OR price below weekly EMA
            if (lips_aligned[i] < teeth_aligned[i]) or (close[i] < ema50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Alligator convergence (Lips > Teeth) OR price above weekly EMA
            if (lips_aligned[i] > teeth_aligned[i]) or (close[i] > ema50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals