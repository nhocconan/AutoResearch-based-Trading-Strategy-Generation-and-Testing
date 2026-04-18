#!/usr/bin/env python3
"""
4h Camarilla Pivot Reversal with 12h Trend Filter
Hypothesis: Price tends to revert from Camarilla pivot levels (R1, S1) during ranging or
weak trending markets, but only when aligned with the 12h trend direction. This avoids
counter-trend trades and captures mean reversion in both bull and bear markets.
Uses volume confirmation to filter false signals. Targets 20-30 trades/year to
minimize fee drag.
"""

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
    
    # Get 12h data for trend filter (once before loop)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # 12h EMA34 for trend filter
    ema34_12h = pd.Series(df_12h['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    # Calculate Camarilla pivot levels for each 4h bar using prior day's OHLC
    # We'll use rolling window to get prior day's values
    # Assuming 16 bars per day (4h timeframe)
    lookback = 16
    
    # Arrays to store pivot levels
    R1 = np.full(n, np.nan)
    S1 = np.full(n, np.nan)
    R2 = np.full(n, np.nan)
    S2 = np.full(n, np.nan)
    
    for i in range(lookback, n):
        # Prior day's OHLC (16 bars back)
        idx = i - lookback
        if idx < 0:
            continue
            
        # Get high, low, close from prior day
        # We'll use the max high, min low, and last close of the prior day window
        day_high = np.max(high[idx:idx+lookback])
        day_low = np.min(low[idx:idx+lookback])
        day_close = close[idx+lookback-1]  # Last bar of prior day
        
        if np.isnan(day_high) or np.isnan(day_low) or np.isnan(day_close):
            continue
            
        # Camarilla equations
        range_val = day_high - day_low
        if range_val <= 0:
            continue
            
        R1[i] = day_close + range_val * 1.1 / 12
        S1[i] = day_close - range_val * 1.1 / 12
        R2[i] = day_close + range_val * 1.1 / 6
        S2[i] = day_close - range_val * 1.1 / 6
    
    # Volume filter: current volume > 1.3x 20-period volume average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(ema34_12h_aligned[i]) or np.isnan(R1[i]) or np.isnan(S1[i]) or 
            np.isnan(R2[i]) or np.isnan(S2[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ok = vol_filter[i]
        trend = ema34_12h_aligned[i]
        
        if position == 0:
            # Long: price at S1 support with volume, in uptrend
            if price <= S1[i] * 1.001 and price >= S1[i] * 0.999 and vol_ok and price > trend:
                signals[i] = 0.25
                position = 1
            # Short: price at R1 resistance with volume, in downtrend
            elif price >= R1[i] * 0.999 and price <= R1[i] * 1.001 and vol_ok and price < trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit if price reaches R1 or trend weakens
            if price >= R1[i] * 0.999 or price < trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit if price reaches S1 or trend weakens
            if price <= S1[i] * 1.001 or price > trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_Pivot_Reversal_12hTrend"
timeframe = "4h"
leverage = 1.0