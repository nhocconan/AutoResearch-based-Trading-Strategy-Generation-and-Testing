#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_13968_12h_donchian20_1w_volume_v1"
timeframe = "12h"
leverage = 1.0

# Hypothesis: 12h Donchian(20) breakout with weekly trend filter and volume confirmation.
# Long when price breaks above 20-period high with volume > 1.5x average and weekly close > weekly open.
# Short when price breaks below 20-period low with volume > 1.5x average and weekly close < weekly open.
# Exit when price returns to the opposite Donchian band or volatility drops.
# Designed for 50-150 total trades over 4 years (12-37/year) to minimize fee drag.
# Works in bull (breaks above with volume) and bear (breaks below with volume).

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data for trend filter ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly trend: bullish if weekly close > weekly open
    weekly_open = df_1w['open'].values
    weekly_close = df_1w['close'].values
    weekly_bullish = weekly_close > weekly_open
    
    # Align weekly trend to 12h timeframe
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish.astype(float))
    
    # 12h data for Donchian channels, ATR, and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    lookback = 20
    donchian_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    donchian_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # ATR for stop loss
    atr = calculate_atr(high, low, close, 14)
    
    # Volume confirmation
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(lookback, 20) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or \
           np.isnan(volume_ma[i]) or np.isnan(atr[i]) or np.isnan(weekly_bullish_aligned[i]):
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
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * 1.5)
        
        # Breakout signals from Donchian channels
        breakout_high = close[i] > donchian_high[i-1]  # break above upper band
        breakdown_low = close[i] < donchian_low[i-1]   # break below lower band
        
        # Exit signals (return to opposite band)
        exit_long = close[i] < donchian_low[i]
        exit_short = close[i] > donchian_high[i]
        
        # Entry signals with weekly trend filter and volume
        long_signal = breakout_high and weekly_bullish_aligned[i] and volume_ok
        short_signal = breakdown_low and (not weekly_bullish_aligned[i]) and volume_ok
        
        # Generate signals
        if position == 0:
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (2.0 * atr[i])
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (2.0 * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long on return to lower band or opposite breakout
            if exit_long:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short on return to upper band or opposite breakout
            if exit_short:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals