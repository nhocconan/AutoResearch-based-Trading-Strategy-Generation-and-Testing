#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_14044_1d_donchian20_1w_ema_vol_v1"
timeframe = "1d"
leverage = 1.0

def calculate_donchian(high, low, period):
    """Calculate Donchian channels: upper = rolling max, lower = rolling min"""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def calculate_ema(arr, period):
    """Calculate Exponential Moving Average"""
    return pd.Series(arr).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_atr(high, low, close, period):
    """Calculate Average True Range"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    return pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load weekly data for EMA trend filter (once before loop)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA(50) for trend filter
    ema_50_1w = calculate_ema(close_1w, 50)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Daily data for Donchian and ATR
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channels (20-period)
    donch_upper, donch_lower = calculate_donchian(high, low, 20)
    
    # Calculate ATR for stop loss and volume filter
    atr = calculate_atr(high, low, close, 14)
    
    # Calculate average volume (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period (max of 50 for weekly EMA, 20 for Donchian, 14 for ATR)
    start = max(50, 20, 14) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(ema_50_1w_aligned[i]) or np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]) or \
           np.isnan(atr[i]) or np.isnan(avg_volume[i]):
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
        
        # Volume confirmation: current volume > 1.5 * average volume
        volume_ok = volume[i] > 1.5 * avg_volume[i]
        
        # Trend filter: price above/below weekly EMA50
        price_above_ema = close[i] > ema_50_1w_aligned[i]
        price_below_ema = close[i] < ema_50_1w_aligned[i]
        
        # Donchian breakout conditions
        breakout_up = close[i] > donch_upper[i-1]  # Break above previous upper band
        breakout_down = close[i] < donch_lower[i-1]  # Break below previous lower band
        
        # Generate signals
        if position == 0:
            # Long: bullish breakout with volume and trend confirmation
            if breakout_up and volume_ok and price_above_ema:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (2.5 * atr[i])
            # Short: bearish breakout with volume and trend confirmation
            elif breakout_down and volume_ok and price_below_ema:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (2.5 * atr[i])
            else:
                signals[i] = 0.0
        # Position management handled in stop checks above
    
    return signals