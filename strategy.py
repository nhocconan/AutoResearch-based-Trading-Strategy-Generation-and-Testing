#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume confirmation and 12h ADX trend filter.
# Long when price breaks above Donchian upper band in bullish regime (ADX>25 and +DI>-DI).
# Short when price breaks below Donchian lower band in bearish regime (ADX>25 and -DI>+DI).
# Works in bull markets (captures breakouts) and bear markets (captures breakdowns).
# Volume > 1.5x average confirms breakout strength.
# ATR-based stop loss limits downside.

name = "exp_13583_4h_donchian20_12h_adx_1d_vol_v1"
timeframe = "4h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
ADX_PERIOD = 14
ADX_THRESHOLD = 25
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_adx(high, low, close, period):
    """Calculate ADX, +DI, -DI"""
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    
    # Directional Movement
    plus_dm = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    minus_dm = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    
    # Smoothing
    tr_smooth = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / tr_smooth
    minus_di = 100 * minus_dm_smooth / tr_smooth
    
    # DX and ADX
    dx = np.where((plus_di + minus_di) != 0, 
                  100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = pd.Series(dx).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    
    return adx, plus_di, minus_di

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 12h data for ADX trend filter ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h ADX for trend regime
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    adx_12h, plus_di_12h, minus_di_12h = calculate_adx(high_12h, low_12h, close_12h, ADX_PERIOD)
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    plus_di_12h_aligned = align_htf_to_ltf(prices, df_12h, plus_di_12h)
    minus_di_12h_aligned = align_htf_to_ltf(prices, df_12h, minus_di_12h)
    
    # Load 1d data for volume confirmation ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d volume moving average
    volume_1d = df_1d['volume'].values
    volume_ma_1d = pd.Series(volume_1d).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    volume_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_1d)
    
    # Calculate 4h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels
    donchian_high = pd.Series(high).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    donchian_low = pd.Series(low).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    
    # ATR for stop loss
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, ADX_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(adx_12h_aligned[i]) or np.isnan(plus_di_12h_aligned[i]) or 
            np.isnan(minus_di_12h_aligned[i]) or np.isnan(volume_ma_1d_aligned[i]) or
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i])):
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
        
        # Volume confirmation (using 1d average)
        volume_ok = volume[i] > (volume_ma_1d_aligned[i] * VOLUME_THRESHOLD)
        
        # Trend regime from 12h ADX and DI
        strong_trend = adx_12h_aligned[i] > ADX_THRESHOLD
        bullish_regime = strong_trend and (plus_di_12h_aligned[i] > minus_di_12h_aligned[i])
        bearish_regime = strong_trend and (minus_di_12h_aligned[i] > plus_di_12h_aligned[i])
        
        # Donchian breakout signals
        breakout_up = close[i] > donchian_high[i-1]  # Break above previous period's high
        breakout_down = close[i] < donchian_low[i-1]  # Break below previous period's low
        
        # Generate signals
        if position == 0:
            if volume_ok and bullish_regime and breakout_up:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif volume_ok and bearish_regime and breakout_down:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long on bearish breakout or trend reversal
            if breakout_down or (not bullish_regime and strong_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            # Exit short on bullish breakout or trend reversal
            if breakout_up or (not bearish_regime and strong_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals