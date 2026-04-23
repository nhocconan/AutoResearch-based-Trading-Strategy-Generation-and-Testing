#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator + Elder Ray combo with 1d EMA50 trend filter and volume confirmation.
Long when Alligator jaws (blue) > teeth (red) > lips (green) AND Elder Bull Power > 0 AND price > 1d EMA50 AND volume > 1.5x 20-period MA.
Short when Alligator jaws < teeth < lips AND Elder Bear Power < 0 AND price < 1d EMA50 AND volume > 1.5x 20-period MA.
Exit when Alligator alignment breaks or Elder Power reverses.
Designed for ~12-25 trades/year with trend-following edge in both bull and bear markets via Alligator's convergence/divergence and Elder Ray's bull/bear power.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Williams Alligator (13,8,5 SMAs with 8,5,3 offsets)
    # Jaw (blue): 13-period SMMA smoothed 8 bars ahead
    # Teeth (red): 8-period SMMA smoothed 5 bars ahead  
    # Lips (green): 5-period SMMA smoothed 3 bars ahead
    def smma(arr, period):
        """Smoothed Moving Average"""
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        result = np.full_like(arr, np.nan, dtype=np.float64)
        # First value is SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CLOSE) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    # Calculate Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Align 1d indicators to 12h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Calculate volume MA (20-period) for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 13, 20)  # need EMA50, Alligator, volume MA20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(jaw_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or 
            np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price > 1d EMA50 = uptrend, price < 1d EMA50 = downtrend
        trend_up = close[i] > ema_50_1d_aligned[i]
        trend_down = close[i] < ema_50_1d_aligned[i]
        
        # Volume filter: 12h volume > 1.5x 20-period MA
        vol_filter = volume[i] > 1.5 * vol_ma_20[i]
        
        # Alligator alignment: jaws > teeth > lips = bullish, jaws < teeth < lips = bearish
        alligator_bullish = jaw_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > lips_aligned[i]
        alligator_bearish = jaw_aligned[i] < teeth_aligned[i] and teeth_aligned[i] < lips_aligned[i]
        
        # Elder Ray confirmation
        elder_bullish = bull_power_aligned[i] > 0
        elder_bearish = bear_power_aligned[i] < 0
        
        if position == 0:
            # Long: Alligator bullish AND Elder bullish AND uptrend AND volume confirmation
            if alligator_bullish and elder_bullish and trend_up and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: Alligator bearish AND Elder bearish AND downtrend AND volume confirmation
            elif alligator_bearish and elder_bearish and trend_down and vol_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: Alligator alignment breaks or Elder Power reverses
            exit_signal = False
            if position == 1:
                exit_signal = not (alligator_bullish and elder_bullish)
            elif position == -1:
                exit_signal = not (alligator_bearish and elder_bearish)
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_WilliamsAlligator_ElderRay_1dEMA50_Trend_VolumeConfirmation"
timeframe = "12h"
leverage = 1.0