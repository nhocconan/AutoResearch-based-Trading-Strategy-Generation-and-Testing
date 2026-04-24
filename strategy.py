#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator with 1d volume spike and choppiness regime filter.
- Primary timeframe: 12h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for volume average and choppiness calculation.
- Williams Alligator: Jaw(13,8), Teeth(8,5), Lips(5,3) SMAs on median price.
- Entry: Long when Lips > Teeth > Jaw (bullish alignment) AND volume > 2.0 * 1d average volume AND Choppiness Index > 61.8 (ranging market).
         Short when Lips < Teeth < Jaw (bearish alignment) AND volume > 2.0 * 1d average volume AND Choppiness Index > 61.8.
- Exit: Opposite Alligator alignment.
- Signal size: 0.25 discrete to minimize fee drag.
- Alligator catches trends after ranging periods.
- Volume confirmation ensures breakout legitimacy.
- Choppiness filter avoids strong trends where Alligator whipsaws.
- Works in both bull and bear markets as it trades range expansions after contractions.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def sma(values, period):
    """Simple Moving Average with min_periods."""
    return pd.Series(values).rolling(window=period, min_periods=period).mean().values

def choppiness_index(high, low, close, period):
    """Choppiness Index: high values indicate ranging market, low values indicate trending."""
    atr_sum = pd.Series(np.zeros(len(high)))
    for i in range(len(high)):
        tr = max(high[i] - low[i], abs(high[i] - close[i-1] if i>0 else 0), abs(low[i] - close[i-1] if i>0 else 0))
        atr_sum.iloc[i] = tr
    atr_sum = atr_sum.rolling(window=period, min_periods=period).sum()
    
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max()
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min()
    
    chop = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(period)
    return chop.values

def generate_signals(prices):
    n = len(prices)
    if n < 60:  # Need sufficient data for calculations
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    median_price = (high + low) / 2
    
    # Calculate 1d volume average for confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:  # Need sufficient data for volume MA
        return np.zeros(n)
    
    vol_ma_20 = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    # Calculate 1d Choppiness Index for regime filter
    if len(df_1d) < 14:  # Need sufficient data for Choppiness
        return np.zeros(n)
    
    chop_14_1d = choppiness_index(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    chop_14_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_14_1d)
    
    # Calculate Williams Alligator from 12h data
    jaw_period, jaw_shift = 13, 8
    teeth_period, teeth_shift = 8, 5
    lips_period, lips_shift = 5, 3
    
    jaw_raw = sma(median_price, jaw_period)
    teeth_raw = sma(median_price, teeth_period)
    lips_raw = sma(median_price, lips_period)
    
    # Shift the smoothed averages forward
    jaw = np.roll(jaw_raw, -jaw_shift)
    teeth = np.roll(teeth_raw, -teeth_shift)
    lips = np.roll(lips_raw, -lips_shift)
    
    # Invalidate shifted values
    jaw[-jaw_shift:] = np.nan
    teeth[-teeth_shift:] = np.nan
    lips[-lips_shift:] = np.nan
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(jaw_shift, teeth_shift, lips_shift, 20, 14)  # Need Alligator shifts, volume MA, chop
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(vol_ma_20_aligned[i]) or np.isnan(chop_14_1d_aligned[i]) or
            np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Exit conditions: opposite Alligator alignment
        if position != 0:
            # Exit long: bearish alignment (Lips < Teeth < Jaw)
            if position == 1:
                if lips[i] < teeth[i] and teeth[i] < jaw[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: bullish alignment (Lips > Teeth > Jaw)
            elif position == -1:
                if lips[i] > teeth[i] and teeth[i] > jaw[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Alligator alignment with volume confirmation and chop filter
        if position == 0:
            # Alligator alignment signals
            bullish_align = lips[i] > teeth[i] and teeth[i] > jaw[i]
            bearish_align = lips[i] < teeth[i] and teeth[i] < jaw[i]
            
            # Volume confirmation: current volume > 2.0 * 20-period average volume (aligned)
            volume_confirm = curr_volume > 2.0 * vol_ma_20_aligned[i] if not np.isnan(vol_ma_20_aligned[i]) else False
            
            # Choppiness filter: Choppiness Index > 61.8 (ranging market)
            chop_filter = chop_14_1d_aligned[i] > 61.8
            
            if bullish_align and volume_confirm and chop_filter:
                signals[i] = 0.25
                position = 1
            elif bearish_align and volume_confirm and chop_filter:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "12h_WilliamsAlligator_1dVolumeSpike_ChopFilter_v1"
timeframe = "12h"
leverage = 1.0