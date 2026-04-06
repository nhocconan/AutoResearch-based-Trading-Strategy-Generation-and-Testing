#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_14038_1d_donchian20_1w_ema_vol_v1"
timeframe = "1d"
leverage = 1.0

def calculate_ema(series, period):
    return pd.Series(series).ewm(span=period, adjust=False, min_periods=period).mean().values

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
    if n < 100:
        return np.zeros(n)
    
    # Load weekly data for EMA trend (once before loop)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate weekly EMA(50) for trend filter
    ema_50_1w = calculate_ema(close_1w, 50)
    
    # Calculate weekly volume average (20-period) for volume filter
    vol_ma_20_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    
    # Align weekly indicators to daily timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    vol_ma_20_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_20_1w)
    
    # Daily data for Donchian channels and ATR
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate daily Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate daily ATR(14) for stop loss
    atr = calculate_atr(high, low, close, 14)
    
    # Daily volume average (20-period) for volume confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period (max of 50 for EMA, 20 for Donchian/volume, 14 for ATR)
    start = max(50, 20, 14) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma_20_1w_aligned[i]) or \
           np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or \
           np.isnan(atr[i]) or np.isnan(vol_ma_20[i]):
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
        
        # Volume confirmation: current volume > 1.5 * average volume (both daily and weekly)
        vol_confirm = (volume[i] > 1.5 * vol_ma_20[i]) and (volume_1w[i] > 1.5 * vol_ma_20_1w_aligned[i]) if i < len(volume_1w) else (volume[i] > 1.5 * vol_ma_20[i])
        
        # Donchian breakout signals with weekly EMA trend filter
        if position == 0:
            # Long: price breaks above Donchian high + above weekly EMA + volume confirmation
            if close[i] > donchian_high[i] and close[i] > ema_50_1w_aligned[i] and vol_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (2.0 * atr[i])
            # Short: price breaks below Donchian low + below weekly EMA + volume confirmation
            elif close[i] < donchian_low[i] and close[i] < ema_50_1w_aligned[i] and vol_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (2.0 * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long on stop or Donchian breakdown
            if close[i] <= stop_price or close[i] < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short on stop or Donchian breakout
            if close[i] >= stop_price or close[i] > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals