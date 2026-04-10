#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d volume spike and chop regime filter
# - Williams Alligator: Jaw (13-period SMMA, offset 8), Teeth (8-period SMMA, offset 5), Lips (5-period SMMA, offset 3)
# - Long: Lips > Teeth > Jaw (bullish alignment) + 1d volume > 2.0x 20-period MA + 1d chop < 61.8 (trending)
# - Short: Lips < Teeth < Jaw (bearish alignment) + 1d volume > 2.0x 20-period MA + 1d chop < 61.8
# - Exit: Alligator lines cross (Lips-Teeth or Teeth-Jaw) OR chop > 61.8 (range regime)
# - Position sizing: 0.25 discrete level
# - Targets ~12-37 trades/year on 12h timeframe. Uses Alligator for trend detection,
#   volume spike confirms institutional participation, chop filter avoids whipsaws in ranging markets.
#   Works in bull/bear: Alligator catches strong trends, chop filter adapts to regime.

name = "12h_1d_alligator_volume_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute HTF data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Calculate 1d volume MA(20) for spike detection
    volume_ma_20_1d = pd.Series(volume_1d).ewm(span=20, min_periods=20, adjust=False).mean().values
    volume_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    
    # Calculate 1d Choppiness Index (CHOP) for regime filter
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]
    tr2[0] = np.abs(high_1d[0] - close_1d[0])
    tr3[0] = np.abs(low_1d[0] - close_1d[0])
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Sum of TR over 14 periods
    sum_tr_14 = pd.Series(tr_1d).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index: CHOP = 100 * log10(sumTR14 / (hh14 - ll14)) / log10(14)
    # Avoid division by zero
    range_14 = hh_14 - ll_14
    chop_1d = np.where(range_14 > 0, 100 * np.log10(sum_tr_14 / range_14) / np.log10(14), 50)
    # Handle cases where sum_tr_14 is 0
    chop_1d = np.where(np.isnan(chop_1d) | np.isinf(chop_1d), 50, chop_1d)
    
    chop_ma_10_1d = pd.Series(chop_1d).ewm(span=10, min_periods=10, adjust=False).mean().values
    chop_ma_10_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_ma_10_1d)
    
    # Williams Alligator: Smoothed Moving Average (SMMA) with offsets
    def smma(values, period):
        """Smoothed Moving Average"""
        if len(values) < period:
            return np.full_like(values, np.nan)
        result = np.full_like(values, np.nan, dtype=float)
        # First value is simple SMA
        result[period-1] = np.mean(values[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (PERIOD-1) + CURRENT_VALUE) / PERIOD
        for i in range(period, len(values)):
            result[i] = (result[i-1] * (period-1) + values[i]) / period
        return result
    
    # Alligator lines with offsets
    jaw_period, jaw_offset = 13, 8   # Jaw: 13-period SMMA, offset 8 bars
    teeth_period, teeth_offset = 8, 5  # Teeth: 8-period SMMA, offset 5 bars
    lips_period, lips_offset = 5, 3    # Lips: 5-period SMMA, offset 3 bars
    
    jaw = smma(close_1d, jaw_period)
    teeth = smma(close_1d, teeth_period)
    lips = smma(close_1d, lips_period)
    
    # Apply offsets (shift right by offset bars)
    jaw = np.roll(jaw, jaw_offset)
    teeth = np.roll(teeth, teeth_offset)
    lips = np.roll(lips, lips_offset)
    # Set offset bars to NaN
    jaw[:jaw_offset] = np.nan
    teeth[:teeth_offset] = np.nan
    lips[:lips_offset] = np.nan
    
    # Align Alligator lines to lower timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    for i in range(60, n):
        # Skip if any required data is invalid
        if (np.isnan(volume_ma_20_1d_aligned[i]) or 
            np.isnan(chop_ma_10_1d_aligned[i]) or 
            np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > 20-period MA
        volume_ma_20 = pd.Series(volume).ewm(span=20, min_periods=20, adjust=False).mean().values
        vol_confirm_12h = volume[i] > volume_ma_20[i]
        
        # 1d volume spike: current volume > 2.0x 20-period MA
        vol_1d_current = align_htf_to_ltf(prices, df_1d, volume_1d)
        vol_spike_1d = vol_1d_current[i] > 2.0 * volume_ma_20_1d_aligned[i]
        
        # Chop regime: CHOP < 61.8 = trending regime (favor trend following)
        chop_regime = chop_ma_10_1d_aligned[i] < 61.8
        
        # Alligator alignment signals
        lips_above_teeth = lips_aligned[i] > teeth_aligned[i]
        teeth_above_jaw = teeth_aligned[i] > jaw_aligned[i]
        lips_below_teeth = lips_aligned[i] < teeth_aligned[i]
        teeth_below_jaw = teeth_aligned[i] < jaw_aligned[i]
        
        bullish_alignment = lips_above_teeth and teeth_above_jaw
        bearish_alignment = lips_below_teeth and teeth_below_jaw
        
        if position == 0:  # Flat - look for Alligator alignment
            # Long entry: Bullish alignment + vol confirm + vol spike + chop regime
            if (bullish_alignment and vol_confirm_12h and 
                vol_spike_1d and chop_regime):
                position = 1
                signals[i] = 0.25
            # Short entry: Bearish alignment + vol confirm + vol spike + chop regime
            elif (bearish_alignment and vol_confirm_12h and 
                  vol_spike_1d and chop_regime):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit: Alligator lines cross OR chop > 61.8 (range regime)
            lips_teeth_cross = (lips_aligned[i] - teeth_aligned[i]) * (lips_aligned[i-1] - teeth_aligned[i-1]) < 0
            teeth_jaw_cross = (teeth_aligned[i] - jaw_aligned[i]) * (teeth_aligned[i-1] - jaw_aligned[i-1]) < 0
            any_cross = lips_teeth_cross or teeth_jaw_cross
            
            if position == 1:  # Long position
                if any_cross or chop_ma_10_1d_aligned[i] >= 61.8:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                if any_cross or chop_ma_10_1d_aligned[i] >= 61.8:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals