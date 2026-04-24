#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator (Jaw/Teeth/Lips) crossover with 1d volume spike and ATR regime filter.
- Primary timeframe: 12h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for Alligator SMAs (5,8,13 period SMAs of median price), volume average and ATR.
- Williams Alligator: Jaw (13-period SMA of median, smoothed 8 bars), Teeth (8-period SMA, smoothed 5 bars), Lips (5-period SMA, smoothed 3 bars).
- Entry: Long when Lips cross above Teeth AND Teeth above Jaw (bullish alignment) AND volume > 2.0 * 20-period average volume AND ATR(14) < ATR(50) (low volatility regime).
         Short when Lips cross below Teeth AND Teeth below Jaw (bearish alignment) AND volume > 2.0 * 20-period average volume AND ATR(14) < ATR(50).
- Exit: Opposite Alligator crossover (Lips cross back below Teeth for longs, above Teeth for shorts).
- Signal size: 0.25 discrete to minimize fee drag.
- Williams Alligator captures trend initiation after consolidation periods.
- Volume confirmation ensures breakout legitimacy.
- ATR regime filter avoids high-volatility choppy markets where signals fail.
- Works in both bull and bear markets as it catches new trends after ranging periods.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def sma(values, period):
    """Calculate Simple Moving Average with proper min_periods."""
    return pd.Series(values).rolling(window=period, min_periods=period).mean().values

def smma(values, period):
    """Calculate Smoothed Moving Average (SMMA) - similar to Wilder's smoothing."""
    if len(values) < period:
        return np.full(len(values), np.nan)
    result = np.full(len(values), np.nan)
    # First value is SMA
    result[period-1] = np.mean(values[:period])
    # Subsequent values: SMMA = (Prev SMMA * (period-1) + Current Value) / period
    for i in range(period, len(values)):
        result[i] = (result[i-1] * (period-1) + values[i]) / period
    return result

def atr(high, low, close, period):
    """Calculate Average True Range with proper min_periods."""
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    close_series = pd.Series(close)
    tr1 = high_series - low_series
    tr2 = abs(h_series - close_series.shift(1))
    tr3 = abs(low_series - close_series.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_values = tr.ewm(span=period, adjust=False, min_periods=period).mean().values
    return atr_values

def generate_signals(prices):
    n = len(prices)
    if n < 60:  # Need sufficient data for calculations
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate median price for Alligator (typical price: (high+low+close)/3)
    median_price = (high + low + close) / 3.0
    
    # Calculate 1d Williams Alligator components
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:  # Need sufficient data for Alligator
        return np.zeros(n)
    
    # 1d median price
    median_price_1d = (df_1d['high'].values + df_1d['low'].values + df_1d['close'].values) / 3.0
    
    # Alligator components: Jaw (13,8), Teeth (8,5), Lips (5,3)
    # Jaw: 13-period SMA smoothed by 8-period SMMA
    sma_13_1d = sma(median_price_1d, 13)
    jaw_1d = smma(sma_13_1d, 8)
    
    # Teeth: 8-period SMA smoothed by 5-period SMMA
    sma_8_1d = sma(median_price_1d, 8)
    teeth_1d = smma(sma_8_1d, 5)
    
    # Lips: 5-period SMA smoothed by 3-period SMMA
    sma_5_1d = sma(median_price_1d, 5)
    lips_1d = smma(sma_5_1d, 3)
    
    # Align Alligator lines to 12h timeframe
    jaw_1d_aligned = align_htf_to_ltf(prices, df_1d, jaw_1d)
    teeth_1d_aligned = align_htf_to_ltf(prices, df_1d, teeth_1d)
    lips_1d_aligned = align_htf_to_ltf(prices, df_1d, lips_1d)
    
    # Calculate 1d volume average for confirmation (20-period)
    if len(df_1d) < 20:
        return np.zeros(n)
    
    vol_ma_20 = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    # Calculate 1d ATR for regime filter
    if len(df_1d) < 50:
        return np.zeros(n)
    
    atr_14_1d = atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    atr_50_1d = atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 50)
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    atr_50_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50)  # Need 20 for volume MA, 50 for ATR(50)
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(jaw_1d_aligned[i]) or np.isnan(teeth_1d_aligned[i]) or
            np.isnan(lips_1d_aligned[i]) or np.isnan(vol_ma_20_aligned[i]) or
            np.isnan(atr_14_1d_aligned[i]) or np.isnan(atr_50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        prev_lips = lips_1d_aligned[i-1]
        prev_teeth = teeth_1d_aligned[i-1]
        prev_jaw = jaw_1d_aligned[i-1]
        
        # Exit conditions: Alligator crossover reversal
        if position != 0:
            # Exit long: Lips cross below Teeth (bullish alignment broken)
            if position == 1:
                if lips_1d_aligned[i] < teeth_1d_aligned[i] and prev_lips >= prev_teeth:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: Lips cross above Teeth (bearish alignment broken)
            elif position == -1:
                if lips_1d_aligned[i] > teeth_1d_aligned[i] and prev_lips <= prev_teeth:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Alligator alignment with volume confirmation and ATR regime filter
        if position == 0:
            # Williams Alligator signals
            # Bullish alignment: Lips > Teeth > Jaw
            bullish_alignment = lips_1d_aligned[i] > teeth_1d_aligned[i] and teeth_1d_aligned[i] > jaw_1d_aligned[i]
            # Bearish alignment: Lips < Teeth < Jaw
            bearish_alignment = lips_1d_aligned[i] < teeth_1d_aligned[i] and teeth_1d_aligned[i] < jaw_1d_aligned[i]
            
            # Crossover signals (to avoid whipsaws)
            lips_cross_above_teeth = lips_1d_aligned[i] > teeth_1d_aligned[i] and prev_lips <= prev_teeth
            lips_cross_below_teeth = lips_1d_aligned[i] < teeth_1d_aligned[i] and prev_lips >= prev_teeth
            
            # Volume confirmation: current volume > 2.0 * 20-period average volume (aligned)
            volume_confirm = curr_volume > 2.0 * vol_ma_20_aligned[i] if not np.isnan(vol_ma_20_aligned[i]) else False
            
            # ATR regime filter: ATR(14) < ATR(50) (low volatility regime)
            atr_regime = atr_14_1d_aligned[i] < atr_50_1d_aligned[i]
            
            if bullish_alignment and lips_cross_above_teeth and volume_confirm and atr_regime:
                signals[i] = 0.25
                position = 1
            elif bearish_alignment and lips_cross_below_teeth and volume_confirm and atr_regime:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "12h_Williams_Alligator_1dVolumeSpike_ATRRegime_v1"
timeframe = "12h"
leverage = 1.0