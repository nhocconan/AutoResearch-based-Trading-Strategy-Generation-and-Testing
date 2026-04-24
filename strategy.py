#!/usr/bin/env python3
"""
Hypothesis: 4h Williams Alligator strategy with 1d volume confirmation and choppiness regime filter.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 1d for volume average and choppiness index calculation.
- Williams Alligator: Jaw (13-period SMMA smoothed 8), Teeth (8-period SMMA smoothed 5), Lips (5-period SMMA smoothed 3).
- Entry: Long when Lips > Teeth > Jaw (bullish alignment) AND price > Lips AND volume > 1.5 * 1d average volume AND Choppiness Index < 38.2 (trending regime).
         Short when Lips < Teeth < Jaw (bearish alignment) AND price < Lips AND volume > 1.5 * 1d average volume AND Choppiness Index < 38.2.
- Exit: Opposite Alligator alignment or Choppiness Index > 61.8 (choppy regime).
- Signal size: 0.25 discrete to minimize fee drag.
- Alligator identifies trend direction and alignment.
- Volume confirmation ensures trend legitimacy.
- Choppiness filter avoids ranging markets where Alligator whipsaws.
- Works in both bull and bear markets by capturing strong trends while avoiding chop.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(series, period):
    """Smoothed Moving Average (SMMA) with proper min_periods."""
    series = pd.Series(series)
    sma = series.rolling(window=period, min_periods=period).mean()
    # SMMA: first value is SMA, then recursive smoothing
    smma_values = np.full(len(series), np.nan)
    if len(series) >= period:
        smma_values[period-1] = sma.iloc[period-1]
        for i in range(period, len(series)):
            smma_values[i] = (smma_values[i-1] * (period-1) + series.iloc[i]) / period
    return smma_values

def choppiness_index(high, low, close, period):
    """Choppiness Index: measures whether market is choppy (ranging) or not (trending).
    Values > 61.8 = choppy/ranging, < 38.2 = trending."""
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    close_series = pd.Series(close)
    
    # True Range
    tr1 = high_series - low_series
    tr2 = abs(high_series - close_series.shift(1))
    tr3 = abs(low_series - close_series.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Sum of TR over period
    tr_sum = tr.rolling(window=period, min_periods=period).sum()
    
    # Highest high and lowest low over period
    hh = high_series.rolling(window=period, min_periods=period).max()
    ll = low_series.rolling(window=period, min_periods=period).min()
    
    # Choppiness Index formula
    ci = 100 * np.log10(tr_sum / (hh - ll)) / np.log10(period)
    # Handle division by zero when hh == ll
    ci = np.where((hh - ll) == 0, 100, ci)
    return ci.values

def generate_signals(prices):
    n = len(prices)
    if n < 80:  # Need sufficient data for calculations
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
    
    # Calculate 1d Choppiness Index for regime filter
    if len(df_1d) < 14:  # Need sufficient data for choppy index
        return np.zeros(n)
    
    chop_14_1d = choppiness_index(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    chop_14_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_14_1d)
    
    # Calculate Williams Alligator from 4h data
    # Jaw: 13-period SMMA smoothed 8
    jaw_raw = smma(close, 13)
    jaw = smma(jaw_raw, 8) if not np.all(np.isnan(jaw_raw)) else np.full_like(close, np.nan)
    
    # Teeth: 8-period SMMA smoothed 5
    teeth_raw = smma(close, 8)
    teeth = smma(teeth_raw, 5) if not np.all(np.isnan(teeth_raw)) else np.full_like(close, np.nan)
    
    # Lips: 5-period SMMA smoothed 3
    lips_raw = smma(close, 5)
    lips = smma(lips_raw, 3) if not np.all(np.isnan(lips_raw)) else np.full_like(close, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 50  # Enough warmup for all SMMA smoothing
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(vol_ma_20_aligned[i]) or np.isnan(chop_14_1d_aligned[i]) or
            np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Exit conditions: opposite Alligator alignment or choppy regime
        if position != 0:
            # Exit conditions
            lips_teeth_jaw = lips[i] > teeth[i] > jaw[i]  # Bullish alignment
            lips_teeth_jaw_bear = lips[i] < teeth[i] < jaw[i]  # Bearish alignment
            choppy_regime = chop_14_1d_aligned[i] > 61.8  # Choppy market
            
            if position == 1:
                # Exit long: bearish alignment OR choppy regime
                if not lips_teeth_jaw or choppy_regime:
                    signals[i] = 0.0
                    position = 0
                    continue
            elif position == -1:
                # Exit short: bullish alignment OR choppy regime
                if lips_teeth_jaw or choppy_regime:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Alligator alignment with volume confirmation and trending regime
        if position == 0:
            # Alligator alignments
            bullish_alignment = lips[i] > teeth[i] > jaw[i]
            bearish_alignment = lips[i] < teeth[i] < jaw[i]
            
            # Price relative to Lips
            price_above_lips = curr_close > lips[i]
            price_below_lips = curr_close < lips[i]
            
            # Volume confirmation: current volume > 1.5 * 20-period average volume (aligned)
            volume_confirm = curr_volume > 1.5 * vol_ma_20_aligned[i] if not np.isnan(vol_ma_20_aligned[i]) else False
            
            # Trending regime: Choppiness Index < 38.2
            trending_regime = chop_14_1d_aligned[i] < 38.2
            
            if bullish_alignment and price_above_lips and volume_confirm and trending_regime:
                signals[i] = 0.25
                position = 1
            elif bearish_alignment and price_below_lips and volume_confirm and trending_regime:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "4h_WilliamsAlligator_1dVolumeConfirm_ChopFilter_v1"
timeframe = "4h"
leverage = 1.0