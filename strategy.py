#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d Elder Ray trend filter and volume spike confirmation.
# Enter long when Alligator jaws (13-period SMMA) cross above teeth (8-period SMMA) with 1d Elder Bull Power > 0 and volume > 1.5x 20-bar average.
# Enter short when Alligator jaws cross below teeth with 1d Elder Bear Power < 0 and volume > 1.5x 20-bar average.
# Exit when Alligator jaws cross back over teeth in opposite direction.
# Uses discrete position sizing (0.25) to limit drawdown and reduce fee churn.
# Target: 50-150 total trades over 4 years (12-37/year).
# Williams Alligator identifies trend initiation; 1d Elder Ray ensures higher timeframe alignment; volume spike filters weak signals.
# Works in both bull (strong breakouts) and bear (strong breakdowns).

name = "12h_Williams_Alligator_1dElderRay_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Elder Ray
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d EMA13 for Elder Ray
    close_1d = df_1d['close'].values
    ema_13 = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Align EMA13 to 12h
    ema_13_aligned = align_htf_to_ltf(prices, df_1d, ema_13)
    
    # Calculate Elder Bull Power and Bear Power
    bull_power = high - ema_13_aligned  # using 12h high for consistency
    bear_power = low - ema_13_aligned   # using 12h low for consistency
    
    # Williams Alligator on 12h: SMMA (Smoothed Moving Average)
    # Jaw: 13-period SMMA, Teeth: 8-period SMMA, Lips: 5-period SMMA
    def smma(values, period):
        """Smoothed Moving Average"""
        if len(values) < period:
            return np.full_like(values, np.nan)
        result = np.full_like(values, np.nan)
        # First value is SMA
        result[period-1] = np.mean(values[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (PERIOD-1) + CURRENT_VALUE) / PERIOD
        for i in range(period, len(values)):
            result[i] = (result[i-1] * (period-1) + values[i]) / period
        return result
    
    jaw = smma(close, 13)  # Jaw
    teeth = smma(close, 8)  # Teeth
    lips = smma(close, 5)   # Lips (not used in signals but calculated for completeness)
    
    # Volume confirmation: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 13)  # Ensure sufficient history for volume MA and Alligator
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(volume_ma_20[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        # Alligator signals: Jaw crossing Teeth
        jaw_above_teeth = jaw[i] > teeth[i]
        jaw_below_teeth = jaw[i] < teeth[i]
        
        # Previous bar crossover detection
        if i > start_idx:
            prev_jaw_above_teeth = jaw[i-1] > teeth[i-1]
            prev_jaw_below_teeth = jaw[i-1] < teeth[i-1]
            
            # Bullish crossover: Jaw crosses above Teeth
            bullish_cross = jaw_above_teeth and not prev_jaw_above_teeth
            # Bearish crossover: Jaw crosses below Teeth
            bearish_cross = jaw_below_teeth and not prev_jaw_below_teeth
        else:
            bullish_cross = False
            bearish_cross = False
        
        price = close[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long entry: bullish crossover, Elder Bull Power > 0, volume confirm
            if bullish_cross and bull_power[i] > 0 and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: bearish crossover, Elder Bear Power < 0, volume confirm
            elif bearish_cross and bear_power[i] < 0 and vol_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - hold or exit on bearish crossover
            if bearish_cross:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - hold or exit on bullish crossover
            if bullish_cross:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals