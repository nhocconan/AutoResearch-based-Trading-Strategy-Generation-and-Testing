#!/usr/bin/env python3
# Hypothesis: 12h Williams Alligator with 1d trend filter and volume confirmation.
# Williams Alligator consists of three smoothed moving averages (Jaws=13, Teeth=8, Lips=5).
# Long when Lips > Teeth > Jaws (bullish alignment) AND price > 1d EMA50 AND volume > 1.5x average.
# Short when Lips < Teeth < Jaws (bearish alignment) AND price < 1d EMA50 AND volume > 1.5x average.
# Exit when Alligator alignment breaks (Lips crosses Teeth or Teeth crosses Jaws) OR trend reversal.
# Uses 12h timeframe for lower frequency, Williams Alligator for trend strength, 1d EMA for trend filter, volume for confirmation.
# Target: 50-150 total trades over 4 years (12-37/year). Works in bull via trend continuation, bear via faded rallies.

name = "12h_WilliamsAlligator_1dTrend_Volume_v1"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Williams Alligator calculation
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate SMMA (Smoothed Moving Average) for Williams Alligator
    # Jaws: Period 13, Teeth: Period 8, Lips: Period 5
    def smma(data, period):
        """Calculate Smoothed Moving Average"""
        sma = pd.Series(data).rolling(window=period, min_periods=period).mean().values
        smma_vals = np.full_like(data, np.nan, dtype=float)
        smma_vals[period-1] = sma[period-1]
        for i in range(period, len(data)):
            if not np.isnan(smma_vals[i-1]) and not np.isnan(data[i]):
                smma_vals[i] = (smma_vals[i-1] * (period-1) + data[i]) / period
            else:
                smma_vals[i] = np.nan
        return smma_vals
    
    jaws = smma(close_12h, 13)  # Blue line
    teeth = smma(close_12h, 8)   # Red line
    lips = smma(close_12h, 5)    # Green line
    
    # Volume filter: current 12h volume > 1.5x 20-period average
    vol_ma_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    volume_filter_12h = volume_12h > (1.5 * vol_ma_12h)
    
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
        if (np.isnan(lips[i]) or np.isnan(teeth[i]) or np.isnan(jaws[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma_12h[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Lips > Teeth > Jaws (bullish alignment) AND price > 1d EMA50 AND volume confirmation
            if lips[i] > teeth[i] and teeth[i] > jaws[i] and close[i] > ema50_1d_aligned[i] and volume_filter_12h[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Lips < Teeth < Jaws (bearish alignment) AND price < 1d EMA50 AND volume confirmation
            elif lips[i] < teeth[i] and teeth[i] < jaws[i] and close[i] < ema50_1d_aligned[i] and volume_filter_12h[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Alligator alignment breaks (Lips crosses below Teeth) OR trend reversal (price < 1d EMA50)
            if lips[i] <= teeth[i] or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Alligator alignment breaks (Lips crosses above Teeth) OR trend reversal (price > 1d EMA50)
            if lips[i] >= teeth[i] or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals