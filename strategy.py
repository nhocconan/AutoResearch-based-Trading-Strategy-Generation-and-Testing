#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_13994_1h_4h1d_donchian_vol_trend_v1"
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
    if n < 50:
        return np.zeros(n)
    
    # Load 4h and 1d data for trend filters
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h EMA(20) for trend bias
    ema_4h = calculate_ema(df_4h['close'].values, 20)
    
    # Calculate 1d Donchian(20) for long-term structure
    dch_1d_upper, dch_1d_lower = calculate_donchian(df_1d['high'].values, df_1d['low'].values, 20)
    
    # Align 4h EMA and 1d Donchian to 1h timeframe
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    dch_1d_upper_aligned = align_htf_to_ltf(prices, df_1d, dch_1d_upper)
    dch_1d_lower_aligned = align_htf_to_ltf(prices, df_1d, dch_1d_lower)
    
    # 1h data for entry timing
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1h Donchian channels (20-period) for breakout entries
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    
    # ATR for stop loss (14-period)
    atr = calculate_atr(high, low, close, 14)
    
    # Volume confirmation (20-period average)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 8-20 UTC (pre-market to post-close overlap)
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(50, 20, 20) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(ema_4h_aligned[i]) or np.isnan(dch_1d_upper_aligned[i]) or np.isnan(dch_1d_lower_aligned[i]) or \
           np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or np.isnan(volume_ma[i]) or np.isnan(atr[i]):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Session filter: only trade 8-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
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
        
        # Determine trend bias from 4h EMA(20) and 1d Donchian
        bullish_trend = (close[i] > ema_4h_aligned[i]) and (close[i] > dch_1d_lower_aligned[i])
        bearish_trend = (close[i] < ema_4h_aligned[i]) and (close[i] < dch_1d_upper_aligned[i])
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * 1.5)
        
        # 1h Donchian breakout signals (using previous bar's bands)
        breakout_up = close[i] > donchian_upper[i-1]  # break above previous upper band
        breakout_down = close[i] < donchian_lower[i-1]  # break below previous lower band
        
        # Entry signals - only in direction of trend and within session
        long_signal = bullish_trend and volume_ok and breakout_up and in_session
        short_signal = bearish_trend and volume_ok and breakout_down and in_session
        
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