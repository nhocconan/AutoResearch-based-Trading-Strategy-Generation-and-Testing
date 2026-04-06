#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian channel breakout with 1d volatility filter and volume confirmation.
# Long when price breaks above 20-period Donchian high with expanding volume and low volatility regime.
# Short when price breaks below 20-period Donchian low with expanding volume and low volatility regime.
# Uses 1d ATR percentile to filter for low volatility environments where breakouts are more reliable.
# Works in both bull and bear markets by capturing genuine breakouts with volume confirmation.

name = "exp_13582_12h_donchian20_1d_vol_vol_v1"
timeframe = "12h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
ATR_PERIOD = 14
VOLATILITY_LOOKBACK = 50
VOLATILITY_PERCENTILE = 30  # Only trade when volatility is below 30th percentile
SIGNAL_SIZE = 0.25
ATR_STOP_MULTIPLIER = 2.0

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_percentile(data, lookback, percentile):
    """Calculate rolling percentile"""
    return pd.Series(data).rolling(window=lookback, min_periods=lookback).apply(
        lambda x: np.percentile(x, percentile), raw=True
    ).values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for volatility filter ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d ATR for volatility regime
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    atr_1d = calculate_atr(high_1d, low_1d, close_1d, ATR_PERIOD)
    atr_1d_percentile = calculate_percentile(atr_1d, VOLATILITY_LOOKBACK, VOLATILITY_PERCENTILE)
    atr_1d_percentile_aligned = align_htf_to_ltf(prices, df_1d, atr_1d_percentile)
    
    # Calculate 12h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels
    donchian_high = pd.Series(high).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    donchian_low = pd.Series(low).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    
    # Volume confirmation
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR for stop loss
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD, VOLATILITY_LOOKBACK) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(atr[i]) or 
            np.isnan(atr_1d_percentile_aligned[i])):
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
        
        # Volatility filter: only trade when 1d ATR is below its percentile (low volatility)
        low_volatility = atr_1d_percentile_aligned[i] > atr_1d[i]
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD)
        
        # Donchian breakout signals
        long_breakout = close[i] > donchian_high[i-1]  # Break above previous high
        short_breakout = close[i] < donchian_low[i-1]  # Break below previous low
        
        # Combine signals with filters
        long_signal = volume_ok and low_volatility and long_breakout
        short_signal = volume_ok and low_volatility and short_breakout
        
        # Generate signals
        if position == 0:
            if long_signal:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif short_signal:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long on Donchian break of opposite side or stop loss
            if close[i] < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            # Exit short on Donchian break of opposite side or stop loss
            if close[i] > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals