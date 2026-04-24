#!/usr/bin/env python3
"""
Hypothesis: 1d Williams Alligator with 1w EMA50 trend filter and volume confirmation.
- Primary timeframe: 1d targeting 30-100 total trades over 4 years (7-25/year).
- HTF: 1w for EMA50 trend filter (bull/bear regime).
- Williams Alligator: Jaw (13-period SMMA, 8-shift), Teeth (8-period SMMA, 5-shift), Lips (5-period SMMA, 3-shift).
- Regime: Price above EMA50 on 1w = bull trend (favor longs), below = bear trend (favor shorts).
- Entry: Long when Lips cross above Teeth AND bull regime AND volume > 1.5 * 20-day average volume.
         Short when Lips cross below Teeth AND bear regime AND volume > 1.5 * 20-day average volume.
- Exit: Opposite Alligator crossover (Lips cross below Teeth for long exit, above for short exit).
- Signal size: 0.25 discrete to minimize fee drag.
- Works in both bull and bear markets by aligning with weekly trend and using Alligator for trend changes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(data, period):
    """Smoothed Moving Average (SMMA)"""
    if len(data) < period:
        return np.full(len(data), np.nan)
    result = np.full(len(data), np.nan)
    sma = np.mean(data[:period])
    result[period-1] = sma
    for i in range(period, len(data)):
        result[i] = (result[i-1] * (period-1) + data[i]) / period
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
    
    # Calculate 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:  # Need sufficient data for EMA50
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate Williams Alligator on 1d timeframe
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:  # Need sufficient data for Alligator (jaw=13)
        return np.zeros(n)
    
    median_price_1d = (df_1d['high'].values + df_1d['low'].values) / 2.0
    
    # Alligator components: Jaw (13,8), Teeth (8,5), Lips (5,3)
    jaw = smma(median_price_1d, 13)
    teeth = smma(median_price_1d, 8)
    lips = smma(median_price_1d, 5)
    
    # Apply shifts: Jaw 8 bars, Teeth 5 bars, Lips 3 bars
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    
    # Align Alligator and EMA50 to 1d timeframe (prices)
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw_shifted)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth_shifted)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips_shifted)
    
    # Calculate 1d volume average for confirmation (20-period)
    if len(df_1d) < 20:
        return np.zeros(n)
    
    vol_ma_20_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(13, 50)  # Need 13 for Alligator jaw, 50 for EMA50
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(jaw_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(vol_ma_20_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Regime filter: bull trend when price above weekly EMA50, bear when below
        bull_regime = close[i] > ema50_1w_aligned[i]
        bear_regime = close[i] < ema50_1w_aligned[i]
        
        # Volume confirmation: current volume > 1.5 * 20-day average volume
        volume_confirm = volume[i] > 1.5 * vol_ma_20_1d_aligned[i] if not np.isnan(vol_ma_20_1d_aligned[i]) else False
        
        # Alligator crossover signals
        lips_above_teeth = lips_aligned[i] > teeth_aligned[i]
        lips_below_teeth = lips_aligned[i] < teeth_aligned[i]
        
        # Previous bar values for crossover detection
        if i > 0:
            prev_lips_above_teeth = lips_aligned[i-1] > teeth_aligned[i-1]
            prev_lips_below_teeth = lips_aligned[i-1] < teeth_aligned[i-1]
        else:
            prev_lips_above_teeth = False
            prev_lips_below_teeth = False
        
        # Bullish crossover: Lips cross above Teeth
        bullish_cross = lips_above_teeth and not prev_lips_above_teeth
        # Bearish crossover: Lips cross below Teeth
        bearish_cross = lips_below_teeth and not prev_lips_below_teeth
        
        # Exit conditions: opposite Alligator crossover
        if position != 0:
            # Exit long: bearish crossover (Lips cross below Teeth)
            if position == 1 and bearish_cross:
                signals[i] = 0.0
                position = 0
                continue
            # Exit short: bullish crossover (Lips cross above Teeth)
            elif position == -1 and bullish_cross:
                signals[i] = 0.0
                position = 0
                continue
        
        # Entry conditions: Alligator crossover with regime and volume filters
        if position == 0:
            # Long: bullish crossover AND bull regime AND volume confirmation
            long_condition = bullish_cross and bull_regime and volume_confirm
            
            # Short: bearish crossover AND bear regime AND volume confirmation
            short_condition = bearish_cross and bear_regime and volume_confirm
            
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