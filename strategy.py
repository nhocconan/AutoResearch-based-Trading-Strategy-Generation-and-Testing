#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h ADX trend strength with 4h Donchian channel breakout.
# ADX > 25 indicates strong trend. Enter only when price breaks Donchian(20) in trend direction.
# Use 4h Donchian for trend direction and 1h for precise entry timing.
# Volume confirmation ensures breakout validity.
# Works in bull markets (buy breakouts) and bear markets (sell breakdowns).

name = "exp_13594_1h_adx_4h_donchian_vol_v1"
timeframe = "1h"
leverage = 1.0

# Parameters
ADX_PERIOD = 14
ADX_THRESHOLD = 25
DONCHIAN_PERIOD = 20
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.20
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def calculate_dmi(high, low, close, period):
    """Calculate +DI, -DI, and ADX"""
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
    
    # Set first values to 0 (no previous)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    tr_smooth = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    
    # Directional Indicators
    plus_di = 100 * dm_plus_smooth / tr_smooth
    minus_di = 100 * dm_minus_smooth / tr_smooth
    
    # DX and ADX
    dx = np.where((plus_di + minus_di) != 0, 
                  100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = pd.Series(dx).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    
    return adx

def calculate_donchian(high, low, period):
    """Calculate Donchian channel upper and lower bands"""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

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
    
    # Load 4h data for Donchian channel and trend filter ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h Donchian channel
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    donchian_upper, donchian_lower = calculate_donchian(high_4h, low_4h, DONCHIAN_PERIOD)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower)
    
    # Calculate 1h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # ADX for trend strength
    adx = calculate_dmi(high, low, close, ADX_PERIOD)
    
    # ATR for stop loss
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Volume MA
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(ADX_PERIOD, DONCHIAN_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(adx[i]) or np.isnan(donchian_upper_aligned[i]) or 
            np.isnan(donchian_lower_aligned[i]) or np.isnan(volume_ma[i])):
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
        
        # Trend strength filter
        strong_trend = adx[i] > ADX_THRESHOLD
        
        # Breakout signals
        breakout_up = close[i] > donchian_upper_aligned[i]
        breakout_down = close[i] < donchian_lower_aligned[i]
        
        # Generate signals
        if position == 0:
            if strong_trend and volume_ok:
                if breakout_up:
                    signals[i] = SIGNAL_SIZE
                    position = 1
                    entry_price = close[i]
                    stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
                elif breakout_down:
                    signals[i] = -SIGNAL_SIZE
                    position = -1
                    entry_price = close[i]
                    stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long on breakdown or stop loss
            if close[i] < donchian_lower_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            # Exit short on breakout or stop loss
            if close[i] > donchian_upper_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals