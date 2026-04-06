#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d/1w regime filter, volume confirmation, and ATR stoploss.
# Long when price breaks above Donchian upper band in bullish regime (1d EMA50 up, 1w close > open).
# Short when price breaks below Donchian lower band in bearish regime (1d EMA50 down, 1w close < open).
# Volume must be > 1.5x 20-period average. Exit on opposite signal or 2*ATR stop.
# Works in bull markets (captures breakouts) and bear markets (captures breakdowns).

name = "exp_13589_4h_donchian20_1d1w_regime_vol_v1"
timeframe = "4h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
REGIME_EMA_PERIOD = 50
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_ema(close, period):
    """Calculate EMA"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_donchian(high, low, period):
    """Calculate Donchian channels"""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d and 1w data for regime filter ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d EMA for trend regime
    close_1d = df_1d['close'].values
    ema_1d = calculate_ema(close_1d, REGIME_EMA_PERIOD)
    ema_1d_slope = np.diff(ema_1d, prepend=ema_1d[0])  # slope approximation
    ema_1d_slope_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_slope)
    
    # Calculate 1w bullish/bearish regime (close > open = bullish, close < open = bearish)
    open_1w = df_1w['open'].values
    close_1w = df_1w['close'].values
    week_bullish = close_1w > open_1w
    week_bearish = close_1w < open_1w
    week_bullish_aligned = align_htf_to_ltf(prices, df_1w, week_bullish)
    week_bearish_aligned = align_htf_to_ltf(prices, df_1w, week_bearish)
    
    # Calculate 4h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels
    upper, lower = calculate_donchian(high, low, DONCHIAN_PERIOD)
    
    # ATR for stop loss
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Volume MA
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, REGIME_EMA_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(ema_1d_slope_aligned[i]) or np.isnan(week_bullish_aligned[i]) or 
            np.isnan(week_bearish_aligned[i]) or np.isnan(upper[i]) or np.isnan(lower[i]) or 
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
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD)
        
        # Regime filters
        bull_regime = ema_1d_slope_aligned[i] > 0 and week_bullish_aligned[i]
        bear_regime = ema_1d_slope_aligned[i] < 0 and week_bearish_aligned[i]
        
        # Donchian breakout signals with regime filter
        long_signal = volume_ok and (close[i] > upper[i]) and bull_regime
        short_signal = volume_ok and (close[i] < lower[i]) and bear_regime
        
        # Generate signals
        if position == 0:
            if long_signal:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif short_signal:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long on breakdown or regime change
            if close[i] < lower[i] or not bull_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            # Exit short on breakout or regime change
            if close[i] > upper[i] or not bear_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals