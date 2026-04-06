#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily Donchian channel breakout with volume confirmation and ATR stoploss
# Works in bull/bear because breakouts capture strong moves, volume filters false breakouts,
# and ATR adapts to volatility. Target: 80-150 trades over 4 years (20-38/year).

name = "exp_12937_4h_donchian20_1d_vol_atr_v1"
timeframe = "4h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
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

def calculate_donchian(high, low, period):
    """Calculate Donchian channel upper and lower bands"""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data ONCE before loop
    df_daily = get_htf_data(prices, '1d')
    
    # Calculate daily Donchian channels
    high_d = df_daily['high'].values
    low_d = df_daily['low'].values
    close_d = df_daily['close'].values
    vol_d = df_daily['volume'].values
    
    donchian_upper, donchian_lower = calculate_donchian(high_d, low_d, DONCHIAN_PERIOD)
    volume_ma_d = pd.Series(vol_d).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    atr_d = calculate_atr(high_d, low_d, close_d, ATR_PERIOD)
    
    # Align to 4h timeframe
    donchian_upper_4h = align_htf_to_ltf(prices, df_daily, donchian_upper)
    donchian_lower_4h = align_htf_to_ltf(prices, df_daily, donchian_lower)
    volume_ma_4h = align_htf_to_ltf(prices, df_daily, volume_ma_d)
    atr_4h = align_htf_to_ltf(prices, df_daily, atr_d)
    
    # Calculate 4h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    volume_ma_4h_local = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    atr_4h_local = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if daily data not available
        if np.isnan(donchian_upper_4h[i]) or np.isnan(donchian_lower_4h[i]) or np.isnan(volume_ma_4h[i]) or np.isnan(atr_4h[i]):
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
        
        # Volume confirmation (use both daily and 4h volume)
        volume_ok_daily = volume_ma_4h[i] > 0 and volume[i] > (volume_ma_4h[i] * VOLUME_THRESHOLD)
        volume_ok_4h = volume_ma_4h_local[i] > 0 and volume[i] > (volume_ma_4h_local[i] * VOLUME_THRESHOLD)
        volume_ok = volume_ok_daily or volume_ok_4h
        
        # Breakout above/below Donchian bands
        breakout_long = volume_ok and close[i] >= donchian_upper_4h[i]
        breakout_short = volume_ok and close[i] <= donchian_lower_4h[i]
        
        # Generate signals
        if position == 0:
            if breakout_long:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr_4h_local[i])
            elif breakout_short:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr_4h_local[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals