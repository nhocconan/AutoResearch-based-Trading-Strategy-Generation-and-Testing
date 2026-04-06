#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h timeframe with 1d trend filter and 4h volume confirmation
# Uses Donchian breakout with strict entry conditions to limit trades
# Works in bull/bear via trend filter + volume confirmation
# Target: 15-37 trades/year (60-150 total over 4 years)

name = "exp_13994_1h_donchian20_1d_4h_vol_v1"
timeframe = "1h"
leverage = 1.0

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
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    ema_1d = calculate_ema(df_1d['close'].values, 50)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Load 4h data for volume confirmation (HTF)
    df_4h = get_htf_data(prices, '4h')
    volume_4h = df_4h['volume'].values
    volume_4h_ma = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    volume_4h_ma_aligned = align_htf_to_ltf(prices, df_4h, volume_4h_ma)
    
    # 1h data for Donchian, ATR, and session
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    
    # ATR for stop loss (14-period)
    atr = calculate_atr(high, low, close, 14)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(50, 20, 20) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(ema_1d_aligned[i]) or np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or \
           np.isnan(volume_4h_ma_aligned[i]) or np.isnan(atr[i]):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Check stops
        if position == 1:  # long position
            # Check stop loss
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # short position
            # Check stop loss
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Determine trend bias from 1d EMA (50)
        bullish_trend = close[i] > ema_1d_aligned[i]
        bearish_trend = close[i] < ema_1d_aligned[i]
        
        # Volume confirmation (4h volume above average)
        volume_ok = volume[i] > (volume_4h_ma_aligned[i] * 1.5)
        
        # Donchian breakout signals (using previous bar's bands)
        breakout_up = close[i] > donchian_upper[i-1]
        breakout_down = close[i] < donchian_lower[i-1]
        
        # Entry signals - only in direction of 1d trend and within session
        long_signal = bullish_trend and volume_ok and breakout_up and in_session[i]
        short_signal = bearish_trend and volume_ok and breakout_down and in_session[i]
        
        # Generate signals
        if position == 0:
            if long_signal:
                signals[i] = 0.20
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (2.0 * atr[i])
            elif short_signal:
                signals[i] = -0.20
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (2.0 * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long on Donchian breakdown or trend change to bearish
            if close[i] < donchian_lower[i] or not bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short on Donchian breakout or trend change to bullish
            if close[i] > donchian_upper[i] or not bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals