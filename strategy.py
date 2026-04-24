#!/usr/bin/env python3
"""
Hypothesis: 4h Williams Alligator with 1d EMA34 trend filter and volume confirmation.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 1d for EMA34 trend direction and Alligator calculation (based on daily median price).
- Williams Alligator: Jaw (13-period SMMA), Teeth (8-period SMMA), Lips (5-period SMMA) of median price.
- Entry: Long when Lips > Teeth > Jaw (bullish alignment) AND price > 1d EMA34 AND volume > 2.0 * 20-period average volume.
         Short when Lips < Teeth < Jaw (bearish alignment) AND price < 1d EMA34 AND volume > 2.0 * 20-period average volume.
- Exit: Opposite Alligator alignment (Lips <= Teeth or Teeth <= Jaw for long exit; Lips >= Teeth or Teeth >= Jaw for short exit).
- Signal size: 0.25 discrete to minimize fee drag.
- Alligator identifies trend absence (sleeping), formation (awakening), and trend (eating). Works in ranging markets via volatility expansion.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(values, period):
    """Smoothed Moving Average (SMMA) with proper min_periods."""
    if len(values) < period:
        return np.full_like(values, np.nan, dtype=float)
    result = np.full_like(values, np.nan, dtype=float)
    # First value is simple moving average
    result[period-1] = np.mean(values[:period])
    # Subsequent values: SMMA = (prev_smma * (period-1) + current_value) / period
    for i in range(period, len(values)):
        result[i] = (result[i-1] * (period-1) + values[i]) / period
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 60:  # Need sufficient data for calculations
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d median price for Alligator
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need sufficient data for Alligator (max period 13)
        return np.zeros(n)
    
    median_price_1d = (df_1d['high'].values + df_1d['low'].values + df_1d['close'].values) / 3.0
    
    # Williams Alligator components (SMMA)
    jaw_1d = smma(median_price_1d, 13)  # Jaw (13-period)
    teeth_1d = smma(median_price_1d, 8)  # Teeth (8-period)
    lips_1d = smma(median_price_1d, 5)   # Lips (5-period)
    
    # Align Alligator components to 4h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw_1d)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth_1d)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips_1d)
    
    # Calculate 1d EMA34 for trend filter
    ema34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate 1d volume average for confirmation (20-period)
    if len(df_1d) < 20:
        return np.zeros(n)
    
    vol_ma_20_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # Need 34 for EMA34, 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Alligator alignment conditions
        bullish_alignment = (lips_aligned[i] > teeth_aligned[i]) and (teeth_aligned[i] > jaw_aligned[i])
        bearish_alignment = (lips_aligned[i] < teeth_aligned[i]) and (teeth_aligned[i] < jaw_aligned[i])
        
        # Exit conditions: opposite Alligator alignment
        if position != 0:
            # Exit long: not bullish alignment (Lips <= Teeth or Teeth <= Jaw)
            if position == 1:
                if not bullish_alignment:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: not bearish alignment (Lips >= Teeth or Teeth >= Jaw)
            elif position == -1:
                if not bearish_alignment:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Alligator with trend filter and volume confirmation
        if position == 0:
            # Volume confirmation: current volume > 2.0 * 20-period average volume
            volume_confirm = curr_volume > 2.0 * vol_ma_20_1d_aligned[i] if not np.isnan(vol_ma_20_1d_aligned[i]) else False
            
            # Long: Bullish alignment AND price > 1d EMA34
            long_condition = bullish_alignment and (curr_close > ema34_1d_aligned[i]) and volume_confirm
            
            # Short: Bearish alignment AND price < 1d EMA34
            short_condition = bearish_alignment and (curr_close < ema34_1d_aligned[i]) and volume_confirm
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "4h_WilliamsAlligator_1dEMA34_Trend_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0