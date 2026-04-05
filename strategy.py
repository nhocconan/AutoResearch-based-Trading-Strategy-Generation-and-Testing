#!/usr/bin/env python3
"""
Experiment #9494: 1h EMA Crossover with 4h/1d Trend Filter and Volume Spike
Hypothesis: Use 4h EMA(50) for trend direction, 1d EMA(200) for long-term trend filter, 
and 1h EMA(12/26) crossover for entry timing. Add volume spike confirmation 
and session filter (08-20 UTC) to reduce noise. Targets 60-150 total trades 
over 4 years (15-38/year) to balance opportunity and cost. Works in bull 
(bullish EMA alignment) and bear (bearish EMA alignment) with entries only 
in trend direction. Volume spike ensures institutional participation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_9494_1h_ema_crossover_4h_1d_trend_filter_v1"
timeframe = "1h"
leverage = 1.0

# Parameters
EMA_FAST = 12
EMA_SLOW = 26
EMA_TREND_4H = 50
EMA_LONG_1D = 200
VOLUME_SPIKE_MULTIPLIER = 2.0
SIGNAL_SIZE = 0.20
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def calculate_ema(close, period):
    """Calculate EMA"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_atr(high, low, close, period):
    """Calculate ATR"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(span=period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    close_4h = df_4h['close'].values
    ema_4h = calculate_ema(close_4h, EMA_TREND_4H)
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    close_1d = df_1d['close'].values
    ema_1d = calculate_ema(close_1d, EMA_LONG_1D)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate LTF indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    ema_fast = calculate_ema(close, EMA_FAST)
    ema_slow = calculate_ema(close, EMA_SLOW)
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Volume moving average for spike detection
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(EMA_SLOW, EMA_TREND_4H, EMA_LONG_1D, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(ema_4h_aligned[i]) or np.isnan(ema_1d_aligned[i]):
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
        
        # Session filter: only trade 08-20 UTC
        if not (8 <= hours[i] <= 20):
            signals[i] = 0.0
            position = 0
            continue
        
        # Volume spike confirmation
        volume_spike = volume[i] > (volume_ma[i] * VOLUME_SPIKE_MULTIPLIER) if not np.isnan(volume_ma[i]) else False
        
        # Trend filters
        bullish_4h = ema_4h_aligned[i] > close[i]  # Price above 4h EMA50 = bullish
        bearish_4h = ema_4h_aligned[i] < close[i]  # Price below 4h EMA50 = bearish
        
        bullish_1d = ema_1d_aligned[i] > close[i]  # Price above 1d EMA200 = long-term bullish
        bearish_1d = ema_1d_aligned[i] < close[i]  # Price below 1d EMA200 = long-term bearish
        
        # EMA crossover signals
        ema_cross_up = ema_fast[i] > ema_slow[i] and ema_fast[i-1] <= ema_slow[i-1]
        ema_cross_down = ema_fast[i] < ema_slow[i] and ema_fast[i-1] >= ema_slow[i-1]
        
        # Entry conditions
        long_entry = ema_cross_up and bullish_4h and bullish_1d and volume_spike
        short_entry = ema_cross_down and bearish_4h and bearish_1d and volume_spike
        
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