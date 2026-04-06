#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_12827_6h_1d_wpr_vol"
timeframe = "6h"
leverage = 1.0

# Parameters
WPR_PERIOD = 14
WPR_OVERBOUGHT = -20
WPR_OVERSOLD = -80
VOLUME_MA_PERIOD = 24
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0
MAX_HOLD_BARS = 48

def calculate_wpr(high, low, close, period):
    """Williams %R"""
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max()
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min()
    wpr = -100 * ((highest_high - close) / (highest_high - lowest_low))
    return wpr.fillna(0).values

def calculate_atr(high, low, close, period):
    """ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data ONCE before loop
    df_daily = get_htf_data(prices, '1d')
    
    # Calculate Williams %R on daily
    high_d = df_daily['high'].values
    low_d = df_daily['low'].values
    close_d = df_daily['close'].values
    wpr_d = calculate_wpr(high_d, low_d, close_d, WPR_PERIOD)
    
    # Align to 6h timeframe
    wpr_aligned = align_htf_to_ltf(prices, df_daily, wpr_d)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0
    entry_price = 0.0
    stop_price = 0.0
    bars_since_entry = 0
    
    start = max(VOLUME_MA_PERIOD, ATR_PERIOD, WPR_PERIOD) + 1
    
    for i in range(start, n):
        bars_since_entry += 1
        
        # Skip if WPR not available
        if np.isnan(wpr_aligned[i]):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        # Check stoploss
        if position == 1:
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
                continue
        elif position == -1:
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
                continue
        
        # Time-based exit
        if bars_since_entry >= MAX_HOLD_BARS:
            signals[i] = 0.0
            position = 0
            bars_since_entry = 0
            continue
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # WPR signals: long when oversold and rising, short when overbought and falling
        wpr_oversold = wpr_aligned[i] <= WPR_OVERSOLD
        wpr_overbought = wpr_aligned[i] >= WPR_OVERBOUGHT
        wpr_rising = i > start and wpr_aligned[i] > wpr_aligned[i-1]
        wpr_falling = i > start and wpr_aligned[i] < wpr_aligned[i-1]
        
        signal_long = volume_ok and wpr_oversold and wpr_rising
        signal_short = volume_ok and wpr_overbought and wpr_falling
        
        # Generate signals
        if position == 0:
            if signal_long:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
                bars_since_entry = 0
            elif signal_short:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
                bars_since_entry = 0
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals