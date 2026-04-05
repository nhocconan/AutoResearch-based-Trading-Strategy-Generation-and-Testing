#!/usr/bin/env python3
"""
Experiment #9710: 1d Donchian Breakout + Weekly Trend + Volume Confirmation
Hypothesis: On the daily timeframe, Donchian(20) breakouts aligned with weekly trend 
(HMA21) and volume spikes provide high-probability trend continuation signals.
This strategy targets 30-100 total trades over 4 years (7-25/year) by requiring 
three concurrent conditions: price breakout, trend alignment, and volume confirmation.
Works in bull markets (breakouts above weekly uptrend) and bear markets 
(breakouts below weekly downtrend) with volatility-based position sizing.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_9710_1d_donchian_breakout_weekly_trend_volume_v1"
timeframe = "1d"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
HMA_PERIOD = 21
VOLUME_SPIKE_MULTIPLIER = 2.0
VOLUME_MA_PERIOD = 20
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_hma(arr, period):
    """Calculate Hull Moving Average"""
    if len(arr) < period:
        return np.full_like(arr, np.nan)
    half = period // 2
    sqrt = int(np.sqrt(period))
    wma2 = pd.Series(arr).ewm(span=half, adjust=False).mean()
    wma1 = pd.Series(arr).ewm(span=period, adjust=False).mean()
    raw_hma = 2 * wma2 - wma1
    hma = pd.Series(raw_hma).ewm(span=sqrt, adjust=False).mean()
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
    
    # Load HTF data ONCE before loop (weekly for trend)
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate weekly HMA21 for trend
    weekly_close = df_weekly['close'].values
    hma_weekly = calculate_hma(weekly_close, HMA_PERIOD)
    hma_weekly_aligned = align_htf_to_ltf(prices, df_weekly, hma_weekly)
    
    # Calculate daily Donchian channels
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period high/low)
    donchian_high = pd.Series(high).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    donchian_low = pd.Series(low).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    
    # Volume moving average for spike detection
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, HMA_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(hma_weekly_aligned[i]):
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
        
        # Breakout conditions
        bullish_breakout = close[i] > donchian_high[i]
        bearish_breakout = close[i] < donchian_low[i]
        
        # Trend alignment (weekly HMA)
        price_above_weekly_trend = close[i] > hma_weekly_aligned[i]
        price_below_weekly_trend = close[i] < hma_weekly_aligned[i]
        
        # Entry conditions
        long_entry = bullish_breakout and price_above_weekly_trend and volume_spike
        short_entry = bearish_breakout and price_below_weekly_trend and volume_spike
        
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
</code>