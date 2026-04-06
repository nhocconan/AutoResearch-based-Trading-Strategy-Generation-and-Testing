#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator with 1d trend filter and volume confirmation.
# Uses Alligator's Jaw/Teeth/Lips to identify trend strength and direction.
# Works in bull markets (teeth above jaw, lips above teeth) and bear markets (opposite).
# Volume ensures institutional participation in breakouts.
# Target: 50-150 total trades over 4 years (12-37/year).

name = "exp_13559_6h_alligator_1d_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
ALLIGATOR_JAW_PERIOD = 13   # Blue line
ALLIGATOR_TEETH_PERIOD = 8  # Red line
ALLIGATOR_LIPS_PERIOD = 5   # Green line
JAW_SHIFT = 8
TEETH_SHIFT = 5
LIPS_SHIFT = 3
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_alligator_lines(data, jaw_period, teeth_period, lips_period, jaw_shift, teeth_shift, lips_shift):
    """Calculate Williams Alligator lines (SMMA with forward shift)"""
    # Calculate SMMA (Smoothed Moving Average)
    def smma(series, period):
        sma = pd.Series(series).rolling(window=period, min_periods=period).mean()
        # Wilder's smoothing: SMMA(t) = (SMMA(t-1) * (period-1) + price(t)) / period
        smma_vals = np.full_like(series, np.nan, dtype=float)
        if len(series) >= period:
            smma_vals[period-1] = sma[period-1]
            for i in range(period, len(series)):
                if not np.isnan(sma[i]):
                    smma_vals[i] = (smma_vals[i-1] * (period-1) + sma[i]) / period
                else:
                    smma_vals[i] = smma_vals[i-1]
        return smma_vals
    
    jaw = smma(data, jaw_period)
    teeth = smma(data, teeth_period)
    lips = smma(data, lips_period)
    
    # Apply forward shift (jaw: 8 bars, teeth: 5 bars, lips: 3 bars)
    jaw = np.roll(jaw, jaw_shift)
    teeth = np.roll(teeth, teeth_shift)
    lips = np.roll(lips, lips_shift)
    
    # Set shifted values to NaN
    jaw[:jaw_shift] = np.nan
    teeth[:teeth_shift] = np.nan
    lips[:lips_shift] = np.nan
    
    return jaw, teeth, lips

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily EMA for trend filter
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Williams Alligator lines
    jaw, teeth, lips = calculate_alligator_lines(
        close, ALLIGATOR_JAW_PERIOD, ALLIGATOR_TEETH_PERIOD, ALLIGATOR_LIPS_PERIOD,
        JAW_SHIFT, TEETH_SHIFT, LIPS_SHIFT
    )
    
    # Volume MA
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(
        ALLIGATOR_JAW_PERIOD + JAW_SHIFT,
        ALLIGATOR_TEETH_PERIOD + TEETH_SHIFT,
        ALLIGATOR_LIPS_PERIOD + LIPS_SHIFT,
        VOLUME_MA_PERIOD,
        ATR_PERIOD
    ) + 1
    
    for i in range(start, n):
        # Skip if indicators not available
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or np.isnan(volume_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD)
        
        # Trend filter: price above/below daily EMA
        uptrend = close[i] > ema_1d_aligned[i]
        downtrend = close[i] < ema_1d_aligned[i]
        
        # Alligator signals: teeth above jaw AND lips above teeth = uptrend
        # teeth below jaw AND lips below teeth = downtrend
        alligator_long = (teeth[i] > jaw[i]) and (lips[i] > teeth[i])
        alligator_short = (teeth[i] < jaw[i]) and (lips[i] < teeth[i])
        
        # Generate signals
        if position == 0:
            if volume_ok and uptrend and alligator_long:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif volume_ok and downtrend and alligator_short:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals