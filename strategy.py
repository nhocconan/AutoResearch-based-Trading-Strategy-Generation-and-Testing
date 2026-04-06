#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour ADX + Williams Alligator combination with volume confirmation.
# ADX > 25 filters for trending markets; Alligator (Jaw/Teeth/Lips) crossovers
# provide entry signals in direction of trend. Volume confirms institutional participation.
# Works in bull markets (uptrend + buy signal) and bear markets (downtrend + sell signal).
# Target: 50-150 total trades over 4 years (12-37/year).

name = "exp_13311_6h_adx_alligator_volume_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
ADX_PERIOD = 14
ALLIGATOR_JAW = 13   # Smoothed with 8-period offset
ALLIGATOR_TEETH = 8  # Smoothed with 5-period offset
ALLIGATOR_LIPS = 5   # Smoothed with 3-period offset
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def calculate_wma(series, period):
    """Weighted Moving Average"""
    weights = np.arange(1, period + 1)
    return np.convolve(series, weights/weights.sum(), mode='same')

def calculate_adx(high, low, close, period):
    """ADX calculation using Wilder's smoothing"""
    plus_dm = np.zeros_like(high)
    minus_dm = np.zeros_like(high)
    tr = np.zeros_like(high)
    
    for i in range(1, len(high)):
        plus_dm[i] = max(high[i] - high[i-1], 0) if high[i] - high[i-1] > low[i-1] - low[i] else 0
        minus_dm[i] = max(low[i-1] - low[i], 0) if low[i-1] - low[i] > high[i] - high[i-1] else 0
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Smooth using Wilder's method (alpha = 1/period)
    atr = np.zeros_like(high)
    atr[period] = np.mean(tr[1:period+1])
    for i in range(period+1, len(high)):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    
    plus_di = np.zeros_like(high)
    minus_di = np.zeros_like(high)
    dx = np.zeros_like(high)
    
    # Smooth DM values
    plus_dm_smooth = np.zeros_like(high)
    minus_dm_smooth = np.zeros_like(high)
    plus_dm_smooth[period] = np.sum(plus_dm[1:period+1])
    minus_dm_smooth[period] = np.sum(minus_dm[1:period+1])
    
    for i in range(period+1, len(high)):
        plus_dm_smooth[i] = plus_dm_smooth[i-1] - (plus_dm_smooth[i-1]/period) + plus_dm[i]
        minus_dm_smooth[i] = minus_dm_smooth[i-1] - (minus_dm_smooth[i-1]/period) + minus_dm[i]
    
    # Calculate DI and DX
    for i in range(period, len(high)):
        if atr[i] != 0:
            plus_di[i] = 100 * plus_dm_smooth[i] / atr[i]
            minus_di[i] = 100 * minus_dm_smooth[i] / atr[i]
            if plus_di[i] + minus_di[i] != 0:
                dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
    
    # ADX is smoothed DX
    adx = np.zeros_like(high)
    adx[2*period] = np.mean(dx[period:2*period+1])
    for i in range(2*period+1, len(high)):
        adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
    
    return adx

def calculate_alligator(close, jaw_period, teeth_period, lips_period):
    """Williams Alligator: SMMA (Smoothed Moving Average) with offsets"""
    # Jaw: SMMA(median, 13) + 8 offset
    jaw = pd.Series(close).rolling(window=jaw_period, min_periods=jaw_period).mean()
    jaw = jaw.shift(8)
    
    # Teeth: SMMA(median, 8) + 5 offset
    teeth = pd.Series(close).rolling(window=teeth_period, min_periods=teeth_period).mean()
    teeth = teeth.shift(5)
    
    # Lips: SMMA(median, 5) + 3 offset
    lips = pd.Series(close).rolling(window=lips_period, min_periods=lips_period).mean()
    lips = lips.shift(3)
    
    return jaw.values, teeth.values, lips.values

def calculate_atr(high, low, close, period):
    """ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = np.zeros_like(high)
    atr[period] = np.mean(tr[1:period+1])
    for i in range(period+1, len(high)):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data ONCE before loop for ADX
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate ADX on daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, ADX_PERIOD)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Alligator on 6h close
    jaw, teeth, lips = calculate_alligator(close, ALLIGATOR_JAW, ALLIGATOR_TEETH, ALLIGATOR_LIPS)
    
    # Volume MA
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(ADX_PERIOD*2, ALLIGATOR_JAW+8, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if ADX not available
        if np.isnan(adx_1d_aligned[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]):
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
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Trend filter: ADX > 25 indicates trending market
        trending = adx_1d_aligned[i] > 25
        
        # Alligator signals: Lips crossing Teeth
        # Bullish: Lips > Teeth > Jaw
        # Bearish: Lips < Teeth < Jaw
        bullish_alignment = lips[i] > teeth[i] and teeth[i] > jaw[i]
        bearish_alignment = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        # Generate signals
        if position == 0:
            if trending and volume_ok and bullish_alignment:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif trending and volume_ok and bearish_alignment:
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