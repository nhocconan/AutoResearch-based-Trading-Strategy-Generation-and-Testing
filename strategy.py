#!/usr/bin/env python3
"""
Hypothesis: 4h Williams Alligator strategy with 1d volume confirmation and chop regime filter.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 1d for volume average and chop calculation.
- Williams Alligator: uses smoothed medians (Jaw=13, Teeth=8, Lips=5) to identify trends.
- Entry: Long when Lips > Teeth > Jaw (bullish alignment) AND volume > 1.5 * 1d average volume AND chop < 38.2 (trending regime).
         Short when Lips < Teeth < Jaw (bearish alignment) AND volume > 1.5 * 1d average volume AND chop < 38.2.
- Exit: Opposite Alligator alignment or chop > 61.8 (choppy regime).
- Signal size: 0.25 discrete to minimize fee drag.
- Alligator catches trends early; volume confirms strength; chop filter avoids false signals in ranging markets.
- Works in both bull and bear markets by capturing trending moves while avoiding chop.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(data, period):
    """Smoothed Moving Average (used in Williams Alligator)."""
    data_series = pd.Series(data)
    smma_values = np.zeros_like(data_series, dtype=float)
    smma_values[:] = np.nan
    if len(data_series) < period:
        return smma_values.values
    # First value is simple SMA
    smma_values[period-1] = data_series.iloc[:period].mean()
    # Subsequent values: (prev_smma * (period-1) + current_data) / period
    for i in range(period, len(data_series)):
        smma_values[i] = (smma_values[i-1] * (period-1) + data_series.iloc[i]) / period
    return smma_values

def chop(high, low, close, period):
    """Choppiness Index: measures whether market is choppy (ranging) or trending."""
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    close_series = pd.Series(close)
    atr_list = []
    for i in range(len(high_series)):
        if i == 0:
            tr = high_series.iloc[i] - low_series.iloc[i]
        else:
            tr = max(
                high_series.iloc[i] - low_series.iloc[i],
                abs(high_series.iloc[i] - close_series.iloc[i-1]),
                abs(low_series.iloc[i] - close_series.iloc[i-1])
            )
        atr_list.append(tr)
    atr_series = pd.Series(atr_list)
    sum_atr = atr_series.rolling(window=period, min_periods=period).sum()
    highest_high = high_series.rolling(window=period, min_periods=period).max()
    lowest_low = low_series.rolling(window=period, min_periods=period).min()
    chop_values = 100 * np.log10(sum_atr / (highest_high - lowest_low)) / np.log10(period)
    return chop_values.values

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need sufficient data for calculations
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d volume average for confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:  # Need sufficient data for volume MA
        return np.zeros(n)
    
    vol_ma_20 = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    # Calculate 1d chop for regime filter
    if len(df_1d) < 14:  # Need sufficient data for chop
        return np.zeros(n)
    
    chop_1d = chop(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Calculate Williams Alligator from 4h data
    # Jaw: 13-period SMMA of median price, shifted 8 bars
    # Teeth: 8-period SMMA of median price, shifted 5 bars
    # Lips: 5-period SMMA of median price, shifted 3 bars
    median_price = (high + low) / 2
    
    jaw_raw = smma(median_price, 13)
    teeth_raw = smma(median_price, 8)
    lips_raw = smma(median_price, 5)
    
    # Apply shifts (Alligator lines are shifted into the future)
    jaw = np.roll(jaw_raw, 8)
    jaw[:8] = np.nan
    teeth = np.roll(teeth_raw, 5)
    teeth[:5] = np.nan
    lips = np.roll(lips_raw, 3)
    lips[:3] = np.nan
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(13+8, 8+5, 5+3, 20, 14)  # Jaw shift, Teeth shift, Lips shift, volume MA, chop
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(vol_ma_20_aligned[i]) or np.isnan(chop_1d_aligned[i]) or
            np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Exit conditions: opposite Alligator alignment or choppy regime
        if position != 0:
            # Exit if chop regime becomes too high (choppy/ranging market)
            if chop_1d_aligned[i] > 61.8:
                signals[i] = 0.0
                position = 0
                continue
            # Exit long: Alligator alignment turns bearish
            elif position == 1:
                if not (lips[i] > teeth[i] > jaw[i]):
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: Alligator alignment turns bullish
            elif position == -1:
                if not (lips[i] < teeth[i] < jaw[i]):
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Alligator alignment with volume confirmation and trending regime
        if position == 0:
            # Alligator alignment signals
            bullish_alignment = lips[i] > teeth[i] > jaw[i]
            bearish_alignment = lips[i] < teeth[i] < jaw[i]
            
            # Volume confirmation: current volume > 1.5 * 20-period average volume (aligned)
            volume_confirm = curr_volume > 1.5 * vol_ma_20_aligned[i] if not np.isnan(vol_ma_20_aligned[i]) else False
            
            # Chop regime filter: chop < 38.2 (trending regime)
            trending_regime = chop_1d_aligned[i] < 38.2
            
            if bullish_alignment and volume_confirm and trending_regime:
                signals[i] = 0.25
                position = 1
            elif bearish_alignment and volume_confirm and trending_regime:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "4h_WilliamsAlligator_1dVolumeSpike_ChopFilter_v1"
timeframe = "4h"
leverage = 1.0