#!/usr/bin/env python3
"""
6h Williams Alligator + 1d EMA50 Trend Filter with Volume Confirmation
Hypothesis: Williams Alligator (Jaw/Teeth/Lips) identifies trend emergence and exhaustion.
In bull markets: long when Lips > Teeth > Jaw + price above 1d EMA50 + volume spike.
In bear markets: short when Lips < Teeth < Jaw + price below 1d EMA50 + volume spike.
Volume confirmation reduces false signals. Target 50-150 trades over 4 years (12-37/year).
ATR-based trailing stop manages risk. Works in both regimes by fading extended moves.
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
    sma = pd.Series(series).rolling(window=period, min_periods=period).mean().values
    smma = np.full_like(series, np.nan)
    smma[period-1] = sma[period-1]
    for i in range(period, len(series)):
        smma[i] = (smma[i-1] * (period-1) + series[i]) / period
    return smma

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    median_price = (high + low) / 2  # Williams Alligator uses median price
    
    # Daily data for EMA50 trend (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    
    # Daily EMA50 trend filter
    ema_50_1d = calculate_ema(df_1d['close'].values, 50)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams Alligator on 6h median price
    jaw = calculate_smma(median_price, 13)  # Blue line
    teeth = calculate_smma(median_price, 8)  # Red line
    lips = calculate_smma(median_price, 5)   # Green line
    
    # Volume confirmation: current volume > 1.8 * 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (vol_ma * 1.8)
    
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
    
    # Start index: need enough for Alligator and EMA
    start_idx = max(50, 30, 13) + 10
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        vol_spike = volume_spike[i]
        
        # Alligator alignment conditions
        bullish_alignment = lips[i] > teeth[i] and teeth[i] > jaw[i]
        bearish_alignment = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        if position == 0:
            # Look for entry signals - require: Alligator alignment + volume spike + daily EMA50 trend alignment
            long_entry = bullish_alignment and vol_spike and (curr_close > ema_50_1d_aligned[i])
            short_entry = bearish_alignment and vol_spike and (curr_close < ema_50_1d_aligned[i])
            
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
            
            # Exit conditions: Alligator reversal, trend change, or ATR trailing stop
            trailing_stop = highest_high_since_entry - 2.5 * atr[i]
            if not bullish_alignment or curr_close < ema_50_1d_aligned[i] or curr_close < trailing_stop:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: update lowest low and check exit conditions
            highest_high_since_entry = max(highest_high_since_entry, curr_high)
            lowest_low_since_entry = min(lowest_low_since_entry, curr_low)
            
            # Exit conditions: Alligator reversal, trend change, or ATR trailing stop
            trailing_stop = lowest_low_since_entry + 2.5 * atr[i]
            if not bearish_alignment or curr_close > ema_50_1d_aligned[i] or curr_close > trailing_stop:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsAlligator_1dEMA50_Trend_VolumeSpike_ATRTrailingStop"
timeframe = "6h"
leverage = 1.0