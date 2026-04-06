#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_13969_4h_donchian20_1d_ema_vol_v1"
timeframe = "4h"
leverage = 1.0

# Hypothesis: 4h Donchian(20) breakout with daily EMA trend filter and volume confirmation.
# Uses daily EMA(50) for trend bias: price above daily EMA = bullish bias, price below = bearish bias.
# Entry on 4h Donchian breakout in direction of daily bias with volume > 1.2x average.
# Exit on Donchian reversal or bias change. Designed for 75-200 total trades over 4 years (19-50/year).

def calculate_ema(close, period):
    """Calculate Exponential Moving Average"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_donchian(high, low, period):
    """Calculate Donchian upper and lower bands"""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load daily data for EMA trend filter ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily EMA(50) for trend bias
    daily_ema = calculate_ema(df_1d['close'].values, 50)
    
    # Align daily EMA to 4h timeframe (use prior day's EMA for bias)
    ema_aligned = align_htf_to_ltf(prices, df_1d, daily_ema)
    
    # 4h data for Donchian, ATR, and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    
    # ATR for stop loss
    atr = calculate_atr(high, low, close, 14)
    
    # Volume confirmation
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(60, 50, 20) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(ema_aligned[i]) or np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or \
           np.isnan(volume_ma[i]) or np.isnan(atr[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check stops
        if position == 1:  # long position
            # Check stop loss
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
            continue
        
        elif position == -1:  # short position
            # Check stop loss
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
            continue
        
        # Determine bias from daily EMA (price vs EMA level)
        bullish_bias = close[i] > ema_aligned[i]  # price above daily EMA = bullish bias
        bearish_bias = close[i] < ema_aligned[i]  # price below daily EMA = bearish bias
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * 1.2)
        
        # Donchian breakout signals
        breakout_up = close[i] > donchian_upper[i-1]  # break above previous upper band
        breakout_down = close[i] < donchian_lower[i-1]  # break below previous lower band
        
        # Entry signals - only in direction of daily bias
        long_signal = bullish_bias and volume_ok and breakout_up
        short_signal = bearish_bias and volume_ok and breakout_down
        
        # Generate signals
        if position == 0:
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (2.0 * atr[i])
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (2.0 * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long on Donchian breakdown or bias change to bearish
            if close[i] < donchian_lower[i] or not bullish_bias:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short on Donchian breakout or bias change to bullish
            if close[i] > donchian_upper[i] or not bearish_bias:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals