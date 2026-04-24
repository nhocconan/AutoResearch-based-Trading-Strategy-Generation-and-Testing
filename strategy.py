#!/usr/bin/env python3
"""
Hypothesis: 1d Williams Alligator with 1w EMA50 trend filter and volume confirmation.
- Primary timeframe: 1d targeting 30-100 total trades over 4 years (7-25/year).
- HTF: 1w for EMA50 trend direction.
- Williams Alligator: Jaw (13-period SMMA smoothed 8), Teeth (8-period SMMA smoothed 5), Lips (5-period SMMA smoothed 3).
- Entry: Long when Lips > Teeth > Jaw (bullish alignment) AND price > 1w EMA50 AND volume > 2.0 * 50-period average volume.
         Short when Lips < Teeth < Jaw (bearish alignment) AND price < 1w EMA50 AND volume > 2.0 * 50-period average volume.
- Exit: Opposite Alligator alignment (Lips <= Teeth for long exit, Lips >= Teeth for short exit).
- Signal size: 0.25 discrete to minimize fee drag.
- Alligator identifies trend presence and direction; strong alignment with higher timeframe trend filters noise.
- Works in bull markets (bullish alignment + uptrend) and bear markets (bearish alignment + downtrend).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(values, period):
    """Smoothed Moving Average (SMMA) with proper min_periods."""
    if len(values) < period:
        return np.full(len(values), np.nan)
    sma = pd.Series(values).rolling(window=period, min_periods=period).mean().values
    smma_vals = np.full(len(values), np.nan)
    smma_vals[period-1] = sma[period-1]
    for i in range(period, len(values)):
        smma_vals[i] = (smma_vals[i-1] * (period-1) + values[i]) / period
    return smma_vals

def ema(values, period):
    """Calculate Exponential Moving Average with proper min_periods."""
    return pd.Series(values).ewm(span=period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 100:  # Need sufficient data for calculations
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d Williams Alligator components
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need sufficient data for SMMA calculations
        return np.zeros(n)
    
    # Calculate SMMA for Alligator components
    # Jaw: 13-period SMMA smoothed 8
    jaw_1d = smma(df_1d['close'].values, 13)
    # Teeth: 8-period SMMA smoothed 5
    teeth_1d = smma(df_1d['close'].values, 8)
    # Lips: 5-period SMMA smoothed 3
    lips_1d = smma(df_1d['close'].values, 5)
    
    # Align Alligator components to 1d timeframe (already 1d, but align for consistency)
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw_1d)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth_1d)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips_1d)
    
    # Calculate 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    ema50_1w = ema(df_1w['close'].values, 50)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate 1d volume average for confirmation (50-period)
    if len(df_1d) < 50:
        return np.zeros(n)
    
    vol_ma_50_1d = pd.Series(df_1d['volume'].values).rolling(window=50, min_periods=50).mean().values
    vol_ma_50_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 50)  # Need 50 for Alligator and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_ma_50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Alligator alignment conditions
        bullish_alignment = lips_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > jaw_aligned[i]
        bearish_alignment = lips_aligned[i] < teeth_aligned[i] and teeth_aligned[i] < jaw_aligned[i]
        
        # Exit conditions: opposite Alligator alignment
        if position != 0:
            # Exit long: not bullish alignment (Lips <= Teeth)
            if position == 1:
                if not bullish_alignment:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: not bearish alignment (Lips >= Teeth)
            elif position == -1:
                if not bearish_alignment:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Alligator alignment with trend filter and volume confirmation
        if position == 0:
            # Volume confirmation: current volume > 2.0 * 50-period average volume
            volume_confirm = curr_volume > 2.0 * vol_ma_50_1d_aligned[i] if not np.isnan(vol_ma_50_1d_aligned[i]) else False
            
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

name = "1d_WilliamsAlligator_1wEMA50_Trend_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0