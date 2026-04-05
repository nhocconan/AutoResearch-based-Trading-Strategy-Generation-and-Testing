#!/usr/bin/env python3
"""
Experiment #9454: 1h trend following with 4h/1d filters for direction, volume confirmation for entry.
Hypothesis: In both bull and bear markets, strong trends persist across timeframes. Using 4h/1d for direction
reduces whipsaw, while 1h provides timely entries. Volume confirmation ensures institutional participation.
Target: 60-150 total trades over 4 years (15-37/year) to minimize fee drag.
Session filter (08-20 UTC) reduces noise outside active hours.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_9454_1h_trend_4h1d_volume_session_v1"
timeframe = "1h"
leverage = 1.0

# Parameters
EMA_FAST = 12
EMA_SLOW = 26
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.20
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

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h EMA for trend filter
    close_4h = df_4h['close'].values
    ema_4h_fast = pd.Series(close_4h).ewm(span=EMA_FAST, adjust=False, min_periods=EMA_FAST).mean().values
    ema_4h_slow = pd.Series(close_4h).ewm(span=EMA_SLOW, adjust=False, min_periods=EMA_SLOW).mean().values
    trend_4h = np.where(ema_4h_fast > ema_4h_slow, 1, -1)  # 1=uptrend, -1=downtrend
    trend_4h_aligned = align_htf_to_ltf(prices, df_4h, trend_4h)
    
    # Calculate 1d EMA for stronger trend filter
    close_1d = df_1d['close'].values
    ema_1d_fast = pd.Series(close_1d).ewm(span=EMA_FAST, adjust=False, min_periods=EMA_FAST).mean().values
    ema_1d_slow = pd.Series(close_1d).ewm(span=EMA_SLOW, adjust=False, min_periods=EMA_SLOW).mean().values
    trend_1d = np.where(ema_1d_fast > ema_1d_slow, 1, -1)  # 1=uptrend, -1=downtrend
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # Calculate LTF indicators (1h)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # EMA for entry timing
    ema_fast = pd.Series(close).ewm(span=EMA_FAST, adjust=False, min_periods=EMA_FAST).mean().values
    ema_slow = pd.Series(close).ewm(span=EMA_SLOW, adjust=False, min_periods=EMA_SLOW).mean().values
    
    # Volume moving average
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour  # Pre-compute once
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(EMA_FAST, EMA_SLOW, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
            
        # Skip if HTF data not available
        if np.isnan(trend_4h_aligned[i]) or np.isnan(trend_1d_aligned[i]):
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
        
        # Determine trend alignment (both 4h and 1d must agree)
        bullish_aligned = (trend_4h_aligned[i] == 1) and (trend_1d_aligned[i] == 1)
        bearish_aligned = (trend_4h_aligned[i] == -1) and (trend_1d_aligned[i] == -1)
        
        # EMA crossover for entry timing
        ema_bullish = ema_fast[i] > ema_slow[i]
        ema_bearish = ema_fast[i] < ema_slow[i]
        
        # Volume confirmation
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Entry conditions
        long_entry = bullish_aligned and ema_bullish and volume_confirmed
        short_entry = bearish_aligned and ema_bearish and volume_confirmed
        
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