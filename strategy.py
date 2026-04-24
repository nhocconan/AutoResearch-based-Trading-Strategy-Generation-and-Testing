#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator with 1w trend filter and volume confirmation.
- Primary timeframe: 12h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1w for EMA50 trend direction, 1d for Williams Alligator calculation (based on daily median price).
- Williams Alligator: Jaw=EMA13(median), Teeth=EMA8(median), Lips=EMA5(median) using 1d data.
- Entry: Long when Lips > Teeth > Jaw (bullish alignment) AND price > 1w EMA50 AND volume > 1.5 * 20-period average volume.
         Short when Lips < Teeth < Jaw (bearish alignment) AND price < 1w EMA50 AND volume > 1.5 * 20-period average volume.
- Exit: Opposite Alligator alignment (Lips <= Teeth for long exit, Lips >= Teeth for short exit).
- Signal size: 0.25 discrete to minimize fee drag.
- Williams Alligator identifies trend presence and direction via smoothed medians; works in ranging and trending markets.
- Weekly trend filter prevents counter-trend trades; volume confirmation avoids false breakouts.
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
    
    # Calculate 1d Williams Alligator components (Jaw, Teeth, Lips)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:  # Need sufficient data for EMA13
        return np.zeros(n)
    
    # Median price = (high + low + close) / 3
    median_price = (df_1d['high'].values + df_1d['low'].values + df_1d['close'].values) / 3.0
    
    # Williams Alligator: Jaw=EMA13, Teeth=EMA8, Lips=EMA5
    jaw_1d = ema(median_price, 13)
    teeth_1d = ema(median_price, 8)
    lips_1d = ema(median_price, 5)
    
    # Align Alligator components to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw_1d)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth_1d)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips_1d)
    
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
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50)  # Need 20 for volume MA, 50 for EMA50
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Williams Alligator alignment conditions
        bullish_alignment = lips_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > jaw_aligned[i]
        bearish_alignment = lips_aligned[i] < teeth_aligned[i] and teeth_aligned[i] < jaw_aligned[i]
        
        # Exit conditions: opposite Alligator alignment
        if position != 0:
            # Exit long: bullish alignment broken (lips <= teeth)
            if position == 1:
                if not bullish_alignment:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: bearish alignment broken (lips >= teeth)
            elif position == -1:
                if not bearish_alignment:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Alligator alignment with trend filter and volume confirmation
        if position == 0:
            # Volume confirmation: current volume > 1.5 * 20-period average volume
            volume_confirm = curr_volume > 1.5 * vol_ma_20_1d_aligned[i] if not np.isnan(vol_ma_20_1d_aligned[i]) else False
            
            # Long: Bullish alignment AND price > 1w EMA50
            long_condition = (bullish_alignment and 
                            curr_close > ema50_1w_aligned[i] and
                            volume_confirm)
            
            # Short: Bearish alignment AND price < 1w EMA50
            short_condition = (bearish_alignment and 
                             curr_close < ema50_1w_aligned[i] and
                             volume_confirm)
            
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

name = "12h_WilliamsAlligator_1wEMA50_Trend_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0