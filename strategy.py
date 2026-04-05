#!/usr/bin/env python3
"""
Experiment #10613: 4h Donchian(20) Breakout + 12h EMA Trend + Volume Spike
Hypothesis: 4-hour Donchian breakouts in direction of 12-hour EMA trend with volume confirmation
capture institutional-grade trend continuation. Works in bull markets (breakouts above EMA) and
bear markets (breakdowns below EMA). Volume filter reduces false breakouts. Target: 75-200 trades.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_10613_4h_donchian20_12h_ema_volume_v1"
timeframe = "4h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VOLUME_SPIKE_MULTIPLIER = 1.5
EMA_TREND_PERIOD = 50
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def calculate_donchian_channels(high, low, period):
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def calculate_ema(close, period):
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_atr(high, low, close, period):
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    return pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 12h data ONCE for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_12h = calculate_ema(close_12h, EMA_TREND_PERIOD)
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Calculate 4h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    donch_upper, donch_lower = calculate_donchian_channels(high, low, DONCHIAN_PERIOD)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0
    entry_price = 0.0
    stop_price = 0.0
    
    start = max(DONCHIAN_PERIOD, EMA_TREND_PERIOD, 20) + 1
    
    for i in range(start, n):
        if np.isnan(ema_12h_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
        
        # Stoploss
        if position == 1 and close[i] <= stop_price:
            signals[i] = 0.0
            position = 0
            continue
        elif position == -1 and close[i] >= stop_price:
            signals[i] = 0.0
            position = 0
            continue
        
        # Volume spike
        vol_spike = volume[i] > (volume_ma[i] * VOLUME_SPIKE_MULTIPLIER) if not np.isnan(volume_ma[i]) else False
        
        # Trend and breakout
        above_ema = close[i] > ema_12h_aligned[i]
        below_ema = close[i] < ema_12h_aligned[i]
        bull_break = close[i] > donch_upper[i] if not np.isnan(donch_upper[i]) else False
        bear_break = close[i] < donch_lower[i] if not np.isnan(donch_lower[i]) else False
        
        long_entry = bull_break and above_ema and vol_spike
        short_entry = bear_break and below_ema and vol_spike
        
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