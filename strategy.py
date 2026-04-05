#!/usr/bin/env python3
"""
Experiment #7994: 1-hour timeframe with 4h/1d filters
Hypothesis: Use 4h and 1d timeframes for signal direction to reduce whipsaw, 
and 1h for precise entry timing. Add session filter (08-20 UTC) to avoid low-volume periods. 
Target 60-150 total trades over 4 years (15-37/year) to minimize fee drag.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7994_1h_4h_1d_trend_session"
timeframe = "1h"
leverage = 1.0

# Parameters
TREND_EMA_FAST = 20
TREND_EMA_SLOW = 50
TREND_EMA_1D = 100
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0
SIGNAL_SIZE = 0.20

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h EMAs for trend
    close_4h = df_4h['close'].values
    ema_4h_fast = pd.Series(close_4h).ewm(span=TREND_EMA_FAST, adjust=False, min_periods=TREND_EMA_FAST).mean().values
    ema_4h_slow = pd.Series(close_4h).ewm(span=TREND_EMA_SLOW, adjust=False, min_periods=TREND_EMA_SLOW).mean().values
    trend_4h = np.where(ema_4h_fast > ema_4h_slow, 1, -1)
    trend_4h_aligned = align_htf_to_ltf(prices, df_4h, trend_4h)
    
    # Calculate 1d EMA for higher timeframe trend
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=TREND_EMA_1D, adjust=False, min_periods=TREND_EMA_1D).mean().values
    trend_1d = np.where(close_1d > ema_1d, 1, -1)
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 1h EMA for entry timing
    ema_fast = pd.Series(close).ewm(span=TREND_EMA_FAST, adjust=False, min_periods=TREND_EMA_FAST).mean().values
    ema_slow = pd.Series(close).ewm(span=TREND_EMA_SLOW, adjust=False, min_periods=TREND_EMA_SLOW).mean().values
    
    # ATR for risk management
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(TREND_EMA_SLOW, TREND_EMA_1D, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                # Maintain position but don't add new signals outside session
                signals[i] = position * SIGNAL_SIZE
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
        bullish_alignment = (trend_4h_aligned[i] == 1) and (trend_1d_aligned[i] == 1)
        bearish_alignment = (trend_4h_aligned[i] == -1) and (trend_1d_aligned[i] == -1)
        
        # 1h EMA crossover for entry timing
        ema_cross_up = (ema_fast[i] > ema_slow[i]) and (ema_fast[i-1] <= ema_slow[i-1])
        ema_cross_down = (ema_fast[i] < ema_slow[i]) and (ema_fast[i-1] >= ema_slow[i-1])
        
        # Entry conditions
        long_entry = bullish_alignment and ema_cross_up
        short_entry = bearish_alignment and ema_cross_down
        
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