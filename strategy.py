#!/usr/bin/env python3
"""
Hypothesis: 4h Williams Alligator + Elder Ray + 1w Trend Filter.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 1w for trend filter (only trade in direction of weekly EMA13 slope).
- Entry: Long when Alligator lips (SMA5) > teeth (SMA8) > jaw (SMA13) AND Elder Bull Power > 0 AND weekly EMA13 slope > 0;
         Short when Alligator lips < teeth < jaw AND Elder Bear Power < 0 AND weekly EMA13 slope < 0.
- Exit: Close-based reversal (opposite signal) or Alligator convergence (lips crosses teeth).
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- Williams Alligator identifies trend alignment; Elder Ray measures bull/bear power; weekly trend filter avoids counter-trend whipsaws.
- Works in bull markets (buy when all aligned up) and bear markets (sell when all aligned down) with weekly filter to avoid false signals.
- Estimated trades: ~120 total over 4 years (~30/year) based on Alligator alignment frequency with weekly filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 13:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_13_1w = pd.Series(close_1w).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema_13_slope_1w = ema_13_1w - np.roll(ema_13_1w, 1)
    ema_13_slope_1w[0] = 0
    
    # Calculate Williams Alligator on primary 4h
    # Jaw: SMA13 of median price
    median_price = (high + low) / 2
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().values
    # Teeth: SMA8 of median price
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().values
    # Lips: SMA5 of median price
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().values
    
    # Calculate Elder Ray on primary 4h
    # Bull Power = High - EMA13
    # Bear Power = Low - EMA13
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Align all indicators to primary 4h timeframe
    jaw_aligned = align_htf_to_ltf(prices, prices, jaw)  # Already LTF
    teeth_aligned = align_htf_to_ltf(prices, prices, teeth)
    lips_aligned = align_htf_to_ltf(prices, prices, lips)
    bull_power_aligned = align_htf_to_ltf(prices, prices, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, prices, bear_power)
    ema_13_slope_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_13_slope_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 13  # Need sufficient data for Alligator and Elder Ray
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or 
            np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or np.isnan(ema_13_slope_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Exit conditions
        if position != 0:
            if position == 1:
                # Exit long: Alligator convergence (lips crosses below teeth) OR weekly trend turns down
                if lips_aligned[i] < teeth_aligned[i] or ema_13_slope_1w_aligned[i] <= 0:
                    signals[i] = 0.0
                    position = 0
                    continue
            elif position == -1:
                # Exit short: Alligator convergence (lips crosses above teeth) OR weekly trend turns up
                if lips_aligned[i] > teeth_aligned[i] or ema_13_slope_1w_aligned[i] >= 0:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions
        bullish_alignment = (lips_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > jaw_aligned[i])
        bearish_alignment = (lips_aligned[i] < teeth_aligned[i] and teeth_aligned[i] < jaw_aligned[i])
        
        bullish_power = bull_power_aligned[i] > 0
        bearish_power = bear_power_aligned[i] < 0
        
        weekly_uptrend = ema_13_slope_1w_aligned[i] > 0
        weekly_downtrend = ema_13_slope_1w_aligned[i] < 0
        
        if position == 0:
            # Check for entry signals
            if bullish_alignment and bullish_power and weekly_uptrend:
                signals[i] = 0.25
                position = 1
            elif bearish_alignment and bearish_power and weekly_downtrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "4h_WilliamsAlligator_ElderRay_1wTrendFilter_v1"
timeframe = "4h"
leverage = 1.0