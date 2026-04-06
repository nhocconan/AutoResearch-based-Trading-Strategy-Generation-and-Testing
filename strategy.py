#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily Donchian(20) breakout with 1-day volume confirmation and ATR stoploss.
# Works in bull/bear markets because breakouts capture strong directional moves,
# volume filters out false breakouts, and ATR stoploss adapts to volatility.
# Target: 100-200 trades over 4 years (25-50/year) to balance opportunity and cost.

name = "exp_12929_4h_donchian20_1d_vol_atr_v1"
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
    """Calculate Donchian channels"""
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
    volume_d = df_daily['volume'].values
    
    upper_d, lower_d = calculate_donchian(high_d, low_d, DONCHIAN_PERIOD)
    
    # Calculate daily volume moving average
    volume_ma_d = pd.Series(volume_d).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # Calculate daily ATR
    atr_d = calculate_atr(high_d, low_d, close_d, ATR_PERIOD)
    
    # Align to 4h timeframe
    upper_d_aligned = align_htf_to_ltf(prices, df_daily, upper_d)
    lower_d_aligned = align_htf_to_ltf(prices, df_daily, lower_d)
    volume_ma_d_aligned = align_htf_to_ltf(prices, df_daily, volume_ma_d)
    atr_d_aligned = align_htf_to_ltf(prices, df_daily, atr_d)
    
    # Calculate 4h indicators for entry timing
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 4h ATR for stoploss (more responsive)
    atr_4h = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if daily data not available
        if np.isnan(upper_d_aligned[i]) or np.isnan(lower_d_aligned[i]) or np.isnan(volume_ma_d_aligned[i]) or np.isnan(atr_d_aligned[i]):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        # Check stoploss using 4h ATR
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
        
        # Volume confirmation (using daily average)
        volume_ok = volume[i] > (volume_ma_d_aligned[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma_d_aligned[i]) else False
        
        # Breakout above upper band or breakdown below lower band
        breakout_long = volume_ok and high[i] >= upper_d_aligned[i]
        breakout_short = volume_ok and low[i] <= lower_d_aligned[i]
        
        # Generate signals
        if position == 0:
            if breakout_long:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr_4h[i])
            elif breakout_short:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr_4h[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals