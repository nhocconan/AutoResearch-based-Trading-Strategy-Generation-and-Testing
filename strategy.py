#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + Elder Ray + Volume Spike with 1d trend filter.
# Enter long when Alligator jaws < teeth < lips (bullish alignment), Elder Bull Power > 0, and volume > 2.0x 20-bar average.
# Enter short when Alligator jaws > teeth > lips (bearish alignment), Elder Bear Power < 0, and volume > 2.0x 20-bar average.
# Exit when Alligator alignment reverses or Elder Power crosses zero.
# Uses 1d EMA50 to filter trades in direction of higher timeframe trend.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.
# Williams Alligator identifies trend phases; Elder Ray measures bull/bear power; volume confirms momentum; 1d EMA50 ensures higher timeframe alignment.

name = "12h_WilliamsAlligator_ElderRay_VolumeSpike_1dEMA50_Trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams Alligator (13,8,5 SMAs with future shifts)
    # Jaws: 13-period SMMA shifted 8 bars
    # Teeth: 8-period SMMA shifted 5 bars
    # Lips: 5-period SMMA shifted 3 bars
    def smma(arr, period):
        """Smoothed Moving Average (SMMA)"""
        if len(arr) < period:
            return np.full_like(arr, np.nan, dtype=float)
        smma_vals = np.full_like(arr, np.nan, dtype=float)
        smma_vals[period-1] = np.mean(arr[:period])
        for i in range(period, len(arr)):
            smma_vals[i] = (smma_vals[i-1] * (period-1) + arr[i]) / period
        return smma_vals
    
    smma_13 = smma(close, 13)
    smma_8 = smma(close, 8)
    smma_5 = smma(close, 5)
    
    jaws = np.roll(smma_13, 8)  # shifted 8 bars future
    teeth = np.roll(smma_8, 5)   # shifted 5 bars future
    lips = np.roll(smma_5, 3)    # shifted 3 bars future
    
    # Volume confirmation: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(jaws[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_ma_20[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        # Alligator alignment
        bullish_alignment = jaws[i] < teeth[i] and teeth[i] < lips[i]
        bearish_alignment = jaws[i] > teeth[i] and teeth[i] > lips[i]
        
        # Elder Ray conditions
        bull_power_pos = bull_power[i] > 0
        bear_power_neg = bear_power[i] < 0
        
        # 1d EMA50 trend filter
        trend_up = close[i] > ema_50_1d_aligned[i]
        trend_down = close[i] < ema_50_1d_aligned[i]
        
        # Exit conditions: Alligator alignment reverses or Elder Power crosses zero
        exit_long = not bullish_alignment or bull_power[i] <= 0
        exit_short = not bearish_alignment or bear_power[i] >= 0
        
        # Handle entries and exits
        if bullish_alignment and bull_power_pos and vol_confirm and trend_up and position <= 0:
            signals[i] = 0.30
            position = 1
        elif bearish_alignment and bear_power_neg and vol_confirm and trend_down and position >= 0:
            signals[i] = -0.30
            position = -1
        elif position == 1 and exit_long:
            signals[i] = 0.0
            position = 0
        elif position == -1 and exit_short:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.30
            elif position == -1:
                signals[i] = -0.30
            else:
                signals[i] = 0.0
    
    return signals