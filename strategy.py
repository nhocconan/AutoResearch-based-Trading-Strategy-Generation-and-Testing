#!/usr/bin/env python3
"""
Experiment #7974: 1-hour trend continuation with 4h/1d filters and session filter.
Hypothesis: In trending markets (4h above/below 200 EMA and 1d above/below 200 EMA),
pullbacks to the 20 EMA on 1h during active sessions (08-20 UTC) offer high-probability
entries. Uses 4h/1d for trend direction, 1h for entry timing. Target: 60-150 total trades.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7974_1h_trend_continuation_4h_1d_session_v1"
timeframe = "1h"
leverage = 1.0

# Parameters
EMA_20_PERIOD = 20
EMA_200_PERIOD = 200
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0
SIGNAL_SIZE = 0.20

def generate_signals(prices):
    n = len(prices)
    if n < 250:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h and 1d 200 EMA for trend filter
    close_4h = df_4h['close'].values
    close_1d = df_1d['close'].values
    
    ema_200_4h = pd.Series(close_4h).ewm(span=EMA_200_PERIOD, adjust=False, min_periods=EMA_200_PERIOD).mean().values
    ema_200_1d = pd.Series(close_1d).ewm(span=EMA_200_PERIOD, adjust=False, min_periods=EMA_200_PERIOD).mean().values
    
    # Trend: 1 = bullish (price > EMA200), -1 = bearish (price < EMA200)
    trend_4h = np.where(close_4h > ema_200_4h, 1, -1)
    trend_1d = np.where(close_1d > ema_200_1d, 1, -1)
    
    # Align to 1h timeframe
    trend_4h_aligned = align_htf_to_ltf(prices, df_4h, trend_4h)
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 20 EMA for pullback entries
    ema_20 = pd.Series(close).ewm(span=EMA_20_PERIOD, adjust=False, min_periods=EMA_20_PERIOD).values
    
    # ATR for stop loss
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(EMA_200_PERIOD, EMA_20_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(trend_4h_aligned[i]) or np.isnan(trend_1d_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
        
        # Check stop loss
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
        
        # Determine if we're in an active session
        in_session = 8 <= hours[i] <= 20
        
        # Determine market bias from 4h and 1d trends (both must agree)
        bullish_bias = (trend_4h_aligned[i] == 1) and (trend_1d_aligned[i] == 1)
        bearish_bias = (trend_4h_aligned[i] == -1) and (trend_1d_aligned[i] == -1)
        
        # Pullback to 20 EMA conditions
        near_ema20 = abs(close[i] - ema_20[i]) < (0.1 * atr[i])  # within 10% of ATR from EMA20
        
        # Entry conditions
        long_entry = bullish_bias and near_ema20 and in_session and (close[i] > ema_20[i])
        short_entry = bearish_bias and near_ema20 and in_session and (close[i] < ema_20[i])
        
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