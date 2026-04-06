#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_12843_4h_12h_donchian_vol_regime"
timeframe = "4h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VOLUME_MA_PERIOD = 10
VOLUME_THRESHOLD = 1.8
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5
CHOP_PERIOD = 14
CHOP_THRESHOLD = 61.8  # >61.8 = ranging (mean revert), < = trending

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_chop(high, low, close, period):
    """Calculate Choppiness Index"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    
    atr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum()
    hh = pd.Series(high).rolling(window=period, min_periods=period).max()
    ll = pd.Series(low).rolling(window=period, min_periods=period).min()
    
    chop = 100 * np.log10(atr_sum / (hh - ll)) / np.log10(period)
    return chop.values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h Donchian channels
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate rolling max/min for Donchian
    high_max = pd.Series(high_12h).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    low_min = pd.Series(low_12h).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    
    # Align to 4h timeframe
    donchian_high = align_htf_to_ltf(prices, df_12h, high_max)
    donchian_low = align_htf_to_ltf(prices, df_12h, low_min)
    
    # Calculate 4h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    chop = calculate_chop(high, low, close, CHOP_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD, CHOP_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if Donchian levels not available
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]):
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
        
        # Trend filter: only trade in trending markets (CHOP < threshold)
        trending_market = chop[i] < CHOP_THRESHOLD if not np.isnan(chop[i]) else False
        
        # Breakout logic
        breakout_long = volume_ok and trending_market and close[i] >= donchian_high[i]
        breakout_short = volume_ok and trending_market and close[i] <= donchian_low[i]
        
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