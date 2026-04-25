#!/usr/bin/env python3
"""
1d Williams Alligator Breakout + 1w EMA50 Trend + Volume Spike with ATR Trailing Stop
Hypothesis: Williams Alligator (jaw/teeth/lips) on 1d captures multi-day trend structure, 
with breakouts beyond the Alligator's lips indicating strong momentum. Combined with 
weekly EMA50 trend filter and volume confirmation, this strategy targets 30-100 trades 
over 4 years (7-25/year) to minimize fee drag. ATR-based trailing stop provides risk 
control in both bull and bear markets by exiting on trend exhaustion or reversal.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ema(series, period):
    """Calculate Exponential Moving Average"""
    if len(series) < period:
        return np.full_like(series, np.nan)
    return pd.Series(series).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_smma(series, period):
    """Calculate Smoothed Moving Average (used in Williams Alligator)"""
    if len(series) < period:
        return np.full_like(series, np.nan)
    # SMMA is EMA with alpha = 1/period
    return pd.Series(series).ewm(alpha=1.0/period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_ = prices['open'].values
    
    # Weekly data for EMA50 trend (loaded ONCE)
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly EMA50 trend filter
    ema_50_1w = calculate_ema(df_1w['close'].values, 50)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Daily data for Williams Alligator (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    
    # Williams Alligator: Jaw (13,8), Teeth (8,5), Lips (5,3) - all SMMA of median price
    median_price = (df_1d['high'] + df_1d['low']) / 2
    jaw = calculate_smma(median_price.values, 13)  # Jaw: 13-period, shifted 8 bars
    teeth = calculate_smma(median_price.values, 8)  # Teeth: 8-period, shifted 5 bars
    lips = calculate_smma(median_price.values, 5)   # Lips: 5-period, shifted 3 bars
    
    # Apply shifts (Alligator is typically shifted forward)
    jaw = np.roll(jaw, 8)
    teeth = np.roll(teeth, 5)
    lips = np.roll(lips, 3)
    # First values become NaN due to roll
    jaw[:8] = np.nan
    teeth[:5] = np.nan
    lips[:3] = np.nan
    
    # Align Alligator components to lower timeframe (1d -> 1d, so no change but for consistency)
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    # ATR for trailing stop (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_high_since_entry = 0.0
    lowest_low_since_entry = 0.0
    
    # Start index: need enough for weekly EMA, Alligator, volume MA, and ATR
    start_idx = max(50, 13, 8, 5, 20, 14) + 10
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        vol_spike = volume_spike[i]
        
        # Alligator conditions: Lips above Teeth above Jaw = bullish alignment
        # Lips below Teeth below Jaw = bearish alignment
        bullish_alignment = lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i]
        bearish_alignment = lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i]
        
        # Breakout conditions: price breaks beyond lips with volume
        breakout_long = curr_close > lips_aligned[i] and bullish_alignment
        breakout_short = curr_close < lips_aligned[i] and bearish_alignment
        
        if position == 0:
            # Look for entry signals - require: Alligator breakout + volume spike + weekly EMA50 trend alignment
            long_entry = breakout_long and vol_spike and (curr_close > ema_50_1w_aligned[i])
            short_entry = breakout_short and vol_spike and (curr_close < ema_50_1w_aligned[i])
            
            if long_entry:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                highest_high_since_entry = curr_high
                lowest_low_since_entry = curr_low
            elif short_entry:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
                highest_high_since_entry = curr_high
                lowest_low_since_entry = curr_low
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: update highest high and check exit conditions
            highest_high_since_entry = max(highest_high_since_entry, curr_high)
            lowest_low_since_entry = min(lowest_low_since_entry, curr_low)
            
            # Exit conditions: 
            # 1. Price retreats below lips (trend weakness)
            # 2. Weekly trend changes (price crosses below weekly EMA50)
            # 3. ATR trailing stop (2.5 * ATR from highest high)
            lips_exit = curr_close < lips_aligned[i]
            trend_exit = curr_close < ema_50_1w_aligned[i]
            trailing_stop = highest_high_since_entry - 2.5 * atr[i]
            stop_exit = curr_close < trailing_stop
            
            if lips_exit or trend_exit or stop_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: update lowest low and check exit conditions
            highest_high_since_entry = max(highest_high_since_entry, curr_high)
            lowest_low_since_entry = min(lowest_low_since_entry, curr_low)
            
            # Exit conditions:
            # 1. Price rises above lips (trend weakness)
            # 2. Weekly trend changes (price crosses above weekly EMA50)
            # 3. ATR trailing stop (2.5 * ATR from lowest low)
            lips_exit = curr_close > lips_aligned[i]
            trend_exit = curr_close > ema_50_1w_aligned[i]
            trailing_stop = lowest_low_since_entry + 2.5 * atr[i]
            stop_exit = curr_close > trailing_stop
            
            if lips_exit or trend_exit or stop_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WilliamsAlligator_Breakout_1wEMA50_Trend_VolumeSpike_ATRTrailingStop"
timeframe = "1d"
leverage = 1.0