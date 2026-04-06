#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Elder Ray (Bull/Bear Power) with ADX regime filter and volume confirmation.
# Elder Ray measures bullish/bearish power relative to EMA13. ADX > 25 indicates trending market.
# In trending markets (ADX > 25), trade in direction of Elder Ray: long when Bull Power > 0,
# short when Bear Power < 0. Volume confirms institutional participation. Works in bull
# (sustained bull power) and bear (sustained bear power) markets by following institutional flow.
# Target: 50-150 total trades over 4 years.

name = "elder_ray_adx_regime_6h_v2"
timeframe = "6h"
leverage = 1.0

# Parameters
EMA_LENGTH = 13
ADX_LENGTH = 14
ADX_THRESHOLD = 25
VOLUME_MA_LENGTH = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def ema(close, period):
    """Calculate EMA with proper Wilder's smoothing"""
    return pd.Series(close).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_adx(high, low, close, period):
    """Calculate ADX (Average Directional Index)"""
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    
    # Directional Movement
    dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    
    # Smooth TR, DM+, DM-
    tr_smooth = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / tr_smooth
    di_minus = 100 * dm_minus_smooth / tr_smooth
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    
    return adx

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data ONCE before loop for Elder Ray and ADX
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily EMA for Elder Ray
    close_1d = df_1d['close'].values
    ema_1d = ema(close_1d, EMA_LENGTH)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate daily ADX for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_adx = df_1d['close'].values
    adx_1d = calculate_adx(high_1d, low_1d, close_1d_adx, ADX_LENGTH)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Elder Ray Components
    bull_power = high - ema_1d_aligned  # Bull Power = High - EMA
    bear_power = low - ema_1d_aligned   # Bear Power = Low - EMA
    
    # Volume MA
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_LENGTH, min_periods=VOLUME_MA_LENGTH).mean().values
    
    # ATR
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(EMA_LENGTH, ADX_LENGTH, VOLUME_MA_LENGTH, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if indicators not available
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(adx_1d_aligned[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(volume_ma[i])):
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
        
        # Regime filter: ADX > 25 indicates trending market
        trending = adx_1d_aligned[i] > ADX_THRESHOLD
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Elder Ray signals in trending markets
        if trending and volume_ok:
            bull_signal = bull_power[i] > 0  # Bullish power positive
            bear_signal = bear_power[i] < 0  # Bearish power negative
        else:
            bull_signal = False
            bear_signal = False
        
        # Generate signals
        if position == 0:
            if bull_signal:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif bear_signal:
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