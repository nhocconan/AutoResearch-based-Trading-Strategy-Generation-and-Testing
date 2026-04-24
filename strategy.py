#!/usr/bin/env python3
"""
Hypothesis: 1d Williams Alligator with 1w EMA50 trend filter and volume confirmation.
- Primary timeframe: 1d targeting 30-100 total trades over 4 years (7-25/year).
- HTF: 1w for EMA50 trend direction.
- Williams Alligator: Jaw (EMA13 of median price, 8-bar shift), Teeth (EMA8 of median price, 5-bar shift), Lips (EMA5 of median price, 3-bar shift).
- Entry: Long when Lips > Teeth > Jaw (bullish alignment) AND price > 1w EMA50 AND volume > 1.5 * 20-period average volume.
         Short when Lips < Teeth < Jaw (bearish alignment) AND price < 1w EMA50 AND volume > 1.5 * 20-period average volume.
- Exit: Opposite Alligator alignment (Lips <= Teeth or Teeth <= Jaw for long exit; Lips >= Teeth or Teeth >= Jaw for short exit).
- Signal size: 0.25 discrete to minimize fee drag.
- Williams Alligator identifies trend presence and direction via smoothed median price EMAs.
- Works in bull markets (bullish alignment) and bear markets (bearish alignment) with 1w trend filter avoiding counter-trend trades.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def ema(values, period):
    """Calculate Exponential Moving Average with proper min_periods."""
    return pd.Series(values).ewm(span=period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 60:  # Need sufficient data for calculations
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate median price for Alligator
    median_price = (high + low) / 2
    
    # Calculate 1d Williams Alligator components (Jaw, Teeth, Lips)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need sufficient data for Alligator (max shift 8)
        return np.zeros(n)
    
    median_price_1d = (df_1d['high'].values + df_1d['low'].values) / 2
    
    # Jaw: EMA13 of median price, 8-bar shift
    jaw_1d = ema(median_price_1d, 13)
    jaw_1d = np.roll(jaw_1d, 8)  # shift right by 8
    jaw_1d[:8] = np.nan  # first 8 values invalid
    
    # Teeth: EMA8 of median price, 5-bar shift
    teeth_1d = ema(median_price_1d, 8)
    teeth_1d = np.roll(teeth_1d, 5)  # shift right by 5
    teeth_1d[:5] = np.nan  # first 5 values invalid
    
    # Lips: EMA5 of median price, 3-bar shift
    lips_1d = ema(median_price_1d, 5)
    lips_1d = np.roll(lips_1d, 3)  # shift right by 3
    lips_1d[:3] = np.nan  # first 3 values invalid
    
    # Align Alligator components to 1d timeframe (no shift needed as we're on 1d)
    jaw_aligned = jaw_1d
    teeth_aligned = teeth_1d
    lips_aligned = lips_1d
    
    # Calculate 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    ema50_1w = ema(df_1w['close'].values, 50)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate 1d volume average for confirmation (20-period)
    if len(df_1d) < 20:
        return np.zeros(n)
    
    vol_ma_20_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = vol_ma_20_1d  # already on 1d
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 30)  # Need 20 for volume MA, 30 for Alligator (with shifts)
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(lips_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(jaw_aligned[i]) or
            np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Alligator alignment conditions
        lips_above_teeth = lips_aligned[i] > teeth_aligned[i]
        teeth_above_jaw = teeth_aligned[i] > jaw_aligned[i]
        lips_below_teeth = lips_aligned[i] < teeth_aligned[i]
        teeth_below_jaw = teeth_aligned[i] < jaw_aligned[i]
        
        bullish_alignment = lips_above_teeth and teeth_above_jaw
        bearish_alignment = lips_below_teeth and teeth_below_jaw
        
        # Exit conditions: opposite Alligator alignment
        if position != 0:
            # Exit long: not bullish alignment
            if position == 1:
                if not bullish_alignment:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: not bearish alignment
            elif position == -1:
                if not bearish_alignment:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Alligator with trend filter and volume confirmation
        if position == 0:
            # Volume confirmation: current volume > 1.5 * 20-period average volume
            volume_confirm = curr_volume > 1.5 * vol_ma_20_1d_aligned[i] if not np.isnan(vol_ma_20_1d_aligned[i]) else False
            
            # Long: Bullish alignment AND price > 1w EMA50
            long_condition = bullish_alignment and curr_close > ema50_1w_aligned[i] and volume_confirm
            
            # Short: Bearish alignment AND price < 1w EMA50
            short_condition = bearish_alignment and curr_close < ema50_1w_aligned[i] and volume_confirm
            
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

name = "1d_WilliamsAlligator_1wEMA50_Trend_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0