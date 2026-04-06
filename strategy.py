#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_14053_4h_donchian20_12h_ema_vol_v1"
timeframe = "4h"
leverage = 1.0

def calculate_ema(arr, period):
    return pd.Series(arr).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_donchian(high, low, period):
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def calculate_atr(high, low, close, period):
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Load 12h data for EMA trend filter (once before loop)
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate EMA50 on 12h for trend filter
    ema_12h = calculate_ema(close_12h, 50)
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # 4h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channels (20-period)
    donch_up, donch_low = calculate_donchian(high, low, 20)
    
    # Calculate ATR (14-period) for stop loss
    atr = calculate_atr(high, low, close, 14)
    
    # Volume average (20-period) for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period (max of 20 for Donchian, 50 for EMA)
    start = max(20, 50) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(donch_up[i]) or np.isnan(donch_low[i]) or \
           np.isnan(ema_12h_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma[i]):
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
                continue
        
        elif position == -1:  # short position
            # Check stop loss
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Volume confirmation: current volume > 1.5x average
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Generate signals
        if position == 0:
            # Long: price breaks above Donchian upper + above 12h EMA + volume confirmation
            if close[i] > donch_up[i] and close[i] > ema_12h_aligned[i] and vol_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (2.0 * atr[i])
            # Short: price breaks below Donchian lower + below 12h EMA + volume confirmation
            elif close[i] < donch_low[i] and close[i] < ema_12h_aligned[i] and vol_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (2.0 * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long on stop or price breaks below Donchian lower
            if close[i] <= stop_price or close[i] < donch_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short on stop or price breaks above Donchian upper
            if close[i] >= stop_price or close[i] > donch_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals