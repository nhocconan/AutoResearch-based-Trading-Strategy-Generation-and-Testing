#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with volume confirmation and ATR stoploss
# Works in bull/bear because breakouts capture strong momentum moves, volume filters false signals,
# and ATR stoploss manages risk in volatile conditions. Target: 60-120 trades over 4 years (15-30/year).

name = "exp_12988_12h_donchian20_vol_atr_v1"
timeframe = "12h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

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
    
    # Load daily data ONCE before loop (for Donchian calculation)
    df_daily = get_htf_data(prices, '1d')
    
    # Calculate daily Donchian channels
    high_d = df_daily['high'].values
    low_d = df_daily['low'].values
    donchian_upper, donchian_lower = calculate_donchian(high_d, low_d, DONCHIAN_PERIOD)
    
    # Align to 12h timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_daily, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_daily, donchian_lower)
    
    # Calculate 12h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(VOLUME_MA_PERIOD, ATR_PERIOD, DONCHIAN_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if Donchian levels not available
        if np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]):
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
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Breakout above upper or below lower Donchian
        breakout_long = volume_ok and close[i] >= donchian_upper_aligned[i]
        breakout_short = volume_ok and close[i] <= donchian_lower_aligned[i]
        
        # Generate signals
        if position == 0:
            if breakout_long:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif breakout_short:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals