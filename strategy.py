#!/usr/bin/env python3
"""
Experiment #9523: 4h Donchian Breakout + HMA Trend + Volume Confirmation + ATR Stoploss.
Hypothesis: Donchian channel breakouts combined with Hull Moving Average trend filtering 
and volume confirmation provide high-probability trend continuation signals. 
Works in bull markets via breakouts above upper band and in bear markets via 
breakdowns below lower band. Volume confirms institutional participation. 
ATR-based stoploss manages risk. Targets 75-200 total trades over 4 years (19-50/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_9523_4h_donchian_breakout_hma_volume_v1"
timeframe = "4h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
HMA_FAST = 9
HMA_SLOW = 21
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5  # 1.5x volume average for confirmation
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5
SIGNAL_SIZE = 0.25

def calculate_hma(series, period):
    """Calculate Hull Moving Average"""
    if len(series) < period:
        return np.full_like(series, np.nan)
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = pd.Series(series).ewm(span=half_period, adjust=False).mean()
    wma_full = pd.Series(series).ewm(span=period, adjust=False).mean()
    raw_hma = 2 * wma_half - wma_full
    hma = pd.Series(raw_hma).ewm(span=sqrt_period, adjust=False).mean()
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
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (12h for trend filter)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h HMA for trend filter
    hma_fast_12h = calculate_hma(df_12h['close'].values, HMA_FAST)
    hma_slow_12h = calculate_hma(df_12h['close'].values, HMA_SLOW)
    hma_trend_12h = hma_fast_12h - hma_slow_12h  # Positive = uptrend, Negative = downtrend
    
    # Align 12h HMA trend to 4h timeframe
    hma_trend_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_trend_12h)
    
    # Calculate 4h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels
    donchian_high = pd.Series(high).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    donchian_low = pd.Series(low).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    
    # Volume moving average for confirmation
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD, HMA_SLOW) + 1
    
    for i in range(start, n):
        # Skip if HMA trend data not available
        if np.isnan(hma_trend_12h_aligned[i]):
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
        
        # Volume confirmation
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Donchian breakout conditions
        breakout_up = close[i] > donchian_high[i]
        breakdown_down = close[i] < donchian_low[i]
        
        # Entry conditions with trend filter
        # Long: breakout above upper band + uptrend (12h HMA fast > slow) + volume
        long_entry = breakout_up and (hma_trend_12h_aligned[i] > 0) and volume_confirmed
        # Short: breakdown below lower band + downtrend (12h HMA fast < slow) + volume
        short_entry = breakdown_down and (hma_trend_12h_aligned[i] < 0) and volume_confirmed
        
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