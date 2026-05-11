#!/usr/bin/env python3
"""
12h_Weekly_OBV_Divergence_v1
Hypothesis: Uses On-Balance Volume (OBV) divergence from price on weekly timeframe to identify exhaustion.
When price makes new high/low but OBV fails to confirm, anticipate reversal. Trades only on 12h timeframe
with weekly OBV divergence as filter. Works in bull/bear markets by catching exhaustion moves.
Target: 15-30 trades/year (60-120 over 4 years) with strict divergence confirmation.
"""

name = "12h_Weekly_OBV_Divergence_v1"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data for OBV calculation
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 20:
        return np.zeros(n)
    
    # 12h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- Weekly OBV ---
    close_weekly = df_weekly['close'].values
    volume_weekly = df_weekly['volume'].values
    
    # Calculate OBV: cumulative volume with sign based on price change
    price_change = np.diff(close_weekly, prepend=close_weekly[0])
    volume_signed = np.where(price_change > 0, volume_weekly,
                            np.where(price_change < 0, -volume_weekly, 0))
    obv = np.cumsum(volume_signed)
    
    # Align weekly OBV to 12h timeframe
    obv_aligned = align_htf_to_ltf(prices, df_weekly, obv)
    
    # --- Price extremes detection on 12h ---
    # Look for new 20-period highs/lows
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # --- Divergence conditions ---
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(lookback, 20)  # Ensure sufficient lookback
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(obv_aligned[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i]):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Current price and OBV
        curr_close = close[i]
        curr_obv = obv_aligned[i]
        
        # Bearish divergence: price makes new high but OBV fails to confirm
        bearish_div = (curr_close >= highest_high[i]) and (curr_obv <= obv_aligned[i-1])
        
        # Bullish divergence: price makes new low but OBV fails to confirm
        bullish_div = (curr_close <= lowest_low[i]) and (curr_obv >= obv_aligned[i-1])
        
        if position == 0:
            # Enter on divergence signals
            if bullish_div:
                signals[i] = 0.25
                position = 1
            elif bearish_div:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: opposite divergence or time-based
            if position == 1:
                # Exit long on bearish divergence or after 3 bars
                exit_signal = bearish_div or (i >= start_idx + 3)
                if exit_signal:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short on bullish divergence or after 3 bars
                exit_signal = bullish_div or (i >= start_idx + 3)
                if exit_signal:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals