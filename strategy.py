#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian channel breakout with daily volume confirmation and volatility filter
# Works in bull/bear because breakouts capture strong momentum moves, volume filters false signals,
# and volatility filter avoids ranging markets. Target: 50-150 total trades over 4 years.

name = "exp_12982_12h_donchian20_1d_vol_volfilt_v1"
timeframe = "12h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
VOLATILITY_PERIOD = 14
VOLATILITY_THRESHOLD = 0.5  # ATR/price ratio threshold
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
    
    # Calculate daily indicators
    high_d = df_daily['high'].values
    low_d = df_daily['low'].values
    close_d = df_daily['close'].values
    volume_d = df_daily['volume'].values
    
    # Daily Donchian channels
    donchian_upper_d, donchian_lower_d = calculate_donchian(high_d, low_d, DONCHIAN_PERIOD)
    
    # Daily ATR for volatility filter
    atr_d = calculate_atr(high_d, low_d, close_d, VOLATILITY_PERIOD)
    volatility_ratio = atr_d / close_d  # ATR as percentage of price
    
    # Daily volume MA
    volume_ma_d = pd.Series(volume_d).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # Align daily indicators to 12h timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_daily, donchian_upper_d)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_daily, donchian_lower_d)
    volatility_ratio_aligned = align_htf_to_ltf(prices, df_daily, volatility_ratio)
    volume_ma_aligned = align_htf_to_ltf(prices, df_daily, volume_ma_d)
    
    # Calculate 12h ATR for stoploss
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    atr_12h = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, VOLUME_MA_PERIOD, VOLATILITY_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if indicators not available
        if np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or \
           np.isnan(volatility_ratio_aligned[i]) or np.isnan(volume_ma_aligned[i]):
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
        
        # Volatility filter: only trade when volatility is above threshold
        vol_filter = volatility_ratio_aligned[i] > VOLATILITY_THRESHOLD
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma_aligned[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma_aligned[i]) else False
        
        # Breakout signals
        breakout_long = vol_filter and volume_ok and close[i] >= donchian_upper_aligned[i]
        breakout_short = vol_filter and volume_ok and close[i] <= donchian_lower_aligned[i]
        
        # Generate signals
        if position == 0:
            if breakout_long:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr_12h[i])
            elif breakout_short:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr_12h[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals