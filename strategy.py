#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h timeframe with 1-day/1-week Williams Alligator and Elder Ray
# Uses Williams Alligator (3 SMAs) for trend direction and Elder Ray (bull/bear power)
# for momentum confirmation. Volume filter confirms institutional participation.
# Designed for 12-37 trades/year to minimize fee drag while capturing trend moves.
# Works in bull/bear markets by using Alligator jaws/teeth/lips to identify trends
# and Elder Ray to measure bull/bear power behind the move.

name = "12h_1d_1w_alligator_elder_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily and weekly data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1d) < 20 or len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate Williams Alligator from daily data
    # Jaw: 13-period SMMA, 8 bars into future
    # Teeth: 8-period SMMA, 5 bars into future
    # Lips: 5-period SMMA, 3 bars into future
    close_1d = df_1d['close'].values
    
    def smma(arr, period):
        """Smoothed Moving Average"""
        result = np.full_like(arr, np.nan, dtype=float)
        if len(arr) < period:
            return result
        # First value is SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (prev * (period-1) + current) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(close_1d, 13)
    teeth = smma(close_1d, 8)
    lips = smma(close_1d, 5)
    
    # Calculate Elder Ray from weekly data
    # Bull Power = High - EMA(13)
    # Bear Power = Low - EMA(13)
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # EMA 13 for weekly
    ema_13 = np.full_like(close_1w, np.nan, dtype=float)
    if len(close_1w) >= 13:
        multiplier = 2 / (13 + 1)
        ema_13[12] = np.mean(close_1w[:13])
        for i in range(13, len(close_1w)):
            ema_13[i] = (close_1w[i] - ema_13[i-1]) * multiplier + ema_13[i-1]
    
    bull_power = high_1w - ema_13
    bear_power = low_1w - ema_13
    
    # Align indicators to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    bull_power_aligned = align_htf_to_ltf(prices, df_1w, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1w, bear_power)
    
    # Volume filter: 20-period average
    vol_ma = np.full_like(volume, np.nan, dtype=float)
    for i in range(19, len(volume)):
        vol_ma[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(1, n):
        # Skip if any required data is invalid
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(bull_power_aligned[i]) or
            np.isnan(bear_power_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume filter: current volume > 1.5 * 20-period average
        vol_filter = volume[i] > 1.5 * vol_ma[i]
        
        # Williams Alligator conditions
        # Uptrend: Lips > Teeth > Jaw
        # Downtrend: Jaw > Teeth > Lips
        uptrend = lips_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > jaw_aligned[i]
        downtrend = jaw_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > lips_aligned[i]
        
        # Elder Ray conditions
        # Strong bull power: positive and increasing
        # Strong bear power: negative and decreasing
        strong_bull = bull_power_aligned[i] > 0 and (i == 1 or bull_power_aligned[i] > bull_power_aligned[i-1])
        strong_bear = bear_power_aligned[i] < 0 and (i == 1 or bear_power_aligned[i] < bear_power_aligned[i-1])
        
        # Entry conditions
        # Long: Uptrend + Strong bull power + Volume
        # Short: Downtrend + Strong bear power + Volume
        long_signal = uptrend and strong_bull and vol_filter
        short_signal = downtrend and strong_bear and vol_filter
        
        # Exit conditions
        # Exit long when trend weakens or bear power appears
        exit_long = not uptrend or strong_bear
        # Exit short when trend weakens or bull power appears
        exit_short = not downtrend or strong_bull
        
        # Update position
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
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals