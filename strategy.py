#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + Elder Ray with 1d EMA50 trend filter and volume confirmation
# Uses Williams Alligator (jaw/teeth/lips) for trend structure, Elder Ray (bull/bear power) for momentum,
# 1d EMA50 for higher timeframe trend alignment, and volume spike for confirmation.
# Designed to work in both bull and bear markets by following the 1d trend and using Alligator for
# trend-following signals with Elder Ray filtering false breakouts.
# Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag on 12h timeframe.
# Prioritizes BTC/ETH performance with SOL as secondary.

name = "12h_WilliamsAlligator_ElderRay_1dEMA50_Trend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter and Elder Ray calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Williams Alligator on 12h data
    # Jaw (blue line): 13-period SMMA, shifted 8 bars forward
    # Teeth (red line): 8-period SMMA, shifted 5 bars forward  
    # Lips (green line): 5-period SMMA, shifted 3 bars forward
    def smma(arr, period):
        """Smoothed Moving Average"""
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        result = np.full_like(arr, np.nan, dtype=float)
        # First value is SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT_PRICE) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(close, 13)  # 13-period SMMA
    teeth = smma(close, 8)  # 8-period SMMA
    lips = smma(close, 5)   # 5-period SMMA
    
    # Shift the lines as per Alligator definition
    jaw = np.roll(jaw, 8)   # Shift jaw 8 bars forward
    teeth = np.roll(teeth, 5) # Shift teeth 5 bars forward
    lips = np.roll(lips, 3)   # Shift lips 3 bars forward
    # First values become NaN after roll
    jaw[:8] = np.nan
    teeth[:5] = np.nan
    lips[:3] = np.nan
    
    # Calculate Elder Ray (Bull Power and Bear Power) on 12h data
    # Bull Power = High - EMA13
    # Bear Power = Low - EMA13
    ema_12 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_12
    bear_power = low - ema_12
    
    # Volume confirmation: 20-period EMA on 12h volume
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start from 100 to have valid indicators
        # Skip if any value is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 2.0 x 20-period EMA (tight to avoid overtrading)
        volume_spike = volume[i] > (2.0 * vol_ema_20[i])
        
        # Alligator trend condition: Lips > Teeth > Jaw (uptrend) or Lips < Teeth < Jaw (downtrend)
        alligator_uptrend = lips[i] > teeth[i] and teeth[i] > jaw[i]
        alligator_downtrend = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        # Elder Ray condition: Bull Power > 0 and Bear Power < 0 for strong momentum
        strong_bull = bull_power[i] > 0 and bear_power[i] < 0
        strong_bear = bull_power[i] < 0 and bear_power[i] > 0
        
        # 1d trend filter: price above/below 1d EMA50
        trend_up = close[i] > ema_50_1d_aligned[i]
        trend_down = close[i] < ema_50_1d_aligned[i]
        
        if position == 0:
            # Long: Alligator uptrend + Elder Ray bull + volume spike + 1d uptrend
            if alligator_uptrend and strong_bull and volume_spike and trend_up:
                signals[i] = 0.25
                position = 1
            # Short: Alligator downtrend + Elder Ray bear + volume spike + 1d downtrend
            elif alligator_downtrend and strong_bear and volume_spike and trend_down:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator turns down OR Elder Ray turns bearish OR 1d trend breaks
            if not alligator_uptrend or not strong_bull or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator turns up OR Elder Ray turns bullish OR 1d trend breaks
            if not alligator_downtrend or not strong_bear or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals