#!/usr/bin/env python3
# Hypothesis: 4h Williams Alligator with 1d trend filter and volume confirmation.
# Williams Alligator: Jaw (13-period SMMA smoothed 8), Teeth (8-period SMMA smoothed 5), Lips (5-period SMMA smoothed 3)
# Long when Lips > Teeth > Jaw (bullish alignment) AND price > 1d EMA50 AND volume > 1.5x average
# Short when Lips < Teeth < Jaw (bearish alignment) AND price < 1d EMA50 AND volume > 1.5x average
# Exit when alignment breaks (Lips crosses Teeth or Teeth crosses Jaw) OR trend reversal
# Uses 4h timeframe for optimal trade frequency, Alligator for trend strength, 1d EMA for trend filter, volume for confirmation.
# Target: 100-200 total trades over 4 years (25-50/year). Works in bull via trend continuation, bear via counter-trend fades.

name = "4h_WilliamsAlligator_1dTrend_Volume_v1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(series, period):
    """Smoothed Moving Average (SMMA) - also called Wilder's Moving Average"""
    if len(series) < period:
        return np.full(len(series), np.nan)
    result = np.full(len(series), np.nan)
    # First value is simple SMA
    result[period-1] = np.mean(series[:period])
    # Subsequent values: SMMA = (Prev SMMA * (period-1) + Current Price) / period
    for i in range(period, len(series)):
        result[i] = (result[i-1] * (period-1) + series[i]) / period
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Williams Alligator calculation
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    volume_4h = df_4h['volume'].values
    
    # Calculate Williams Alligator on 4h close
    # Jaw: 13-period SMMA smoothed 8
    jaw_raw = smma(close_4h, 13)
    jaw = smma(jaw_raw, 8)
    # Teeth: 8-period SMMA smoothed 5
    teeth_raw = smma(close_4h, 8)
    teeth = smma(teeth_raw, 5)
    # Lips: 5-period SMMA smoothed 3
    lips_raw = smma(close_4h, 5)
    lips = smma(lips_raw, 3)
    
    # Volume filter: current 4h volume > 1.5x 20-period average
    vol_ma_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    volume_filter_4h = volume_4h > (1.5 * vol_ma_4h)
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA(50) on 1d close for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient data for all indicators
        # Skip if any required data is NaN
        if (np.isnan(lips[i]) or np.isnan(teeth[i]) or np.isnan(jaw[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma_4h[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Lips > Teeth > Jaw (bullish alignment) AND price > 1d EMA50 AND volume confirmation
            if lips[i] > teeth[i] and teeth[i] > jaw[i] and close[i] > ema50_1d_aligned[i] and volume_filter_4h[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Lips < Teeth < Jaw (bearish alignment) AND price < 1d EMA50 AND volume confirmation
            elif lips[i] < teeth[i] and teeth[i] < jaw[i] and close[i] < ema50_1d_aligned[i] and volume_filter_4h[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Alignment breaks (Lips <= Teeth or Teeth <= Jaw) OR trend reversal (price < 1d EMA50)
            if lips[i] <= teeth[i] or teeth[i] <= jaw[i] or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Alignment breaks (Lips >= Teeth or Teeth >= Jaw) OR trend reversal (price > 1d EMA50)
            if lips[i] >= teeth[i] or teeth[i] >= jaw[i] or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals