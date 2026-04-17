#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h weekly EMA50 + daily pivot R1/S1 breakout with volume filter.
# Uses weekly EMA for trend filter, daily pivot levels for entry, volume for confirmation.
# Designed to work in bull (breakouts with trend) and bear (reversals at pivots).
# Target: 15-30 trades/year to avoid fee drag on 12h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Get daily data for pivot levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly EMA50 for trend filter
    close_1w_series = pd.Series(close_1w)
    ema50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate daily pivot points (standard formula)
    # Pivot = (high + low + close) / 3
    # R1 = 2*Pivot - low
    # S1 = 2*Pivot - high
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    r1_1d = 2.0 * pivot_1d - low_1d
    s1_1d = 2.0 * pivot_1d - high_1d
    
    # Align weekly EMA and daily pivots to 12h
    ema50_12h = align_htf_to_ltf(prices, df_1w, ema50_1w)
    r1_12h = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_12h = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Volume filter: current volume > 1.5 * 20-period average (moderate filter)
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # Need weekly EMA50 and volume MA20
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema50_12h[i]) or 
            np.isnan(r1_12h[i]) or 
            np.isnan(s1_12h[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: moderate volume confirmation
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        
        # Trend filter: price above/below weekly EMA50
        price_above_ema = close[i] > ema50_12h[i]
        price_below_ema = close[i] < ema50_12h[i]
        
        # Price relative to daily pivot levels
        price_above_r1 = close[i] > r1_12h[i]
        price_below_s1 = close[i] < s1_12h[i]
        
        if position == 0:
            # Long: Price breaks above daily R1 with volume and above weekly EMA50
            if (price_above_r1 and price_above_ema and volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below daily S1 with volume and below weekly EMA50
            elif (price_below_s1 and price_below_ema and volume_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price crosses below daily S1 OR below weekly EMA50
            if (close[i] < s1_12h[i]) or (close[i] < ema50_12h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price crosses above daily R1 OR above weekly EMA50
            if (close[i] > r1_12h[i]) or (close[i] > ema50_12h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WeeklyEMA50_DailyPivot_R1S1_Volume"
timeframe = "12h"
leverage = 1.0