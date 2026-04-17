#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla Pivot (R1/S1) breakout with 1d trend filter and volume confirmation.
# Uses 1d Camarilla pivot levels for breakouts, 1d EMA21 for trend filter, volume spike for confirmation.
# Designed to work in bull (upward breakouts with trend) and bear (downward breakouts with trend).
# Target: 20-40 trades/year to avoid fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 21:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot and EMA
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla pivot levels (R1, S1)
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    r1_1d = pivot_1d + (range_1d * 1.1 / 12)
    s1_1d = pivot_1d - (range_1d * 1.1 / 12)
    
    # Calculate 1d EMA21 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema21_1d = close_1d_series.ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Align 1d Camarilla and EMA to 4h
    r1_4h = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_4h = align_htf_to_ltf(prices, df_1d, s1_1d)
    ema21_4h = align_htf_to_ltf(prices, df_1d, ema21_1d)
    
    # Volume filter: current volume > 1.5 * 20-period average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 21  # Need 21-period EMA and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_4h[i]) or 
            np.isnan(s1_4h[i]) or 
            np.isnan(ema21_4h[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: spike > 1.5x average
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        
        # Trend filter: price above/below 1d EMA21
        price_above_ema = close[i] > ema21_4h[i]
        price_below_ema = close[i] < ema21_4h[i]
        
        # Price relative to 1d Camarilla levels
        price_above_r1 = close[i] > r1_4h[i]
        price_below_s1 = close[i] < s1_4h[i]
        
        if position == 0:
            # Long: Price breaks above 1d Camarilla R1 with volume and above 1d EMA21
            if (price_above_r1 and price_above_ema and volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below 1d Camarilla S1 with volume and below 1d EMA21
            elif (price_below_s1 and price_below_ema and volume_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price crosses below 1d Camarilla S1 OR below 1d EMA21
            if (close[i] < s1_4h[i]) or (close[i] < ema21_4h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price crosses above 1d Camarilla R1 OR above 1d EMA21
            if (close[i] > r1_4h[i]) or (close[i] > ema21_4h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1S1_1dEMA21_Volume"
timeframe = "4h"
leverage = 1.0