#!/usr/bin/env python3
"""
Experiment #9517: 4h Donchian(20) breakout + HMA trend + volume confirmation + ATR stoploss.
Hypothesis: Donchian breakouts capture momentum in both bull and bear markets when 
confirmed by HMA trend direction and volume spikes. The 4h timeframe balances 
opportunity with cost efficiency, targeting 75-200 total trades over 4 years.
Works in bull markets (breakouts above upper band) and bear markets (breakdowns below lower band).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_9517_4h_donchian20_hma_vol_v1"
timeframe = "4h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
HMA_FAST = 9
HMA_SLOW = 21
VOLUME_SPIKE_MULTIPLIER = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_hma(series, period):
    """Calculate Hull Moving Average"""
    n = int(period)
    if n < 1:
        return series
    half_n = n // 2
    sqrt_n = int(np.sqrt(n))
    
    wma1 = pd.Series(series).ewm(span=half_n, adjust=False, min_periods=half_n).mean()
    wma2 = pd.Series(series).ewm(span=n, adjust=False, min_periods=n).mean()
    raw_hma = 2 * wma1 - wma2
    hma = pd.Series(raw_hma).ewm(span=sqrt_n, adjust=False, min_periods=sqrt_n).mean()
    return hma.values

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
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
    
    # Load HTF data ONCE before loop (1d for trend filter)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HMA for trend filter
    close_1d = df_1d['close'].values
    hma_1d_fast = calculate_hma(close_1d, HMA_FAST)
    hma_1d_slow = calculate_hma(close_1d, HMA_SLOW)
    hma_1d_bullish = hma_1d_fast > hma_1d_slow
    
    # Align 1d HMA trend to 4h timeframe
    hma_1d_bullish_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_bullish.astype(float))
    
    # Calculate LTF indicators (4h)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels
    donchian_high = pd.Series(high).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    donchian_low = pd.Series(low).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    
    # Volume moving average for spike detection
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, 20) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(hma_1d_bullish_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
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
        
        # Volume spike confirmation
        volume_spike = volume[i] > (volume_ma[i] * VOLUME_SPIKE_MULTIPLIER) if not np.isnan(volume_ma[i]) else False
        
        # Trend filter from 1d HMA
        trend_bullish = hma_1d_bullish_aligned[i] > 0.5
        
        # Breakout conditions
        breakout_up = close[i] > donchian_high[i-1]  # Break above previous period's high
        breakout_down = close[i] < donchian_low[i-1]  # Break below previous period's low
        
        # Entry conditions
        long_entry = trend_bullish and volume_spike and breakout_up
        short_entry = (not trend_bullish) and volume_spike and breakout_down
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif short_entry:
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