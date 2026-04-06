#!/usr/bin/env python3
"""
Experiment #11835: 6h Trix Momentum with Weekly Trend Filter and Volume Confirmation
Hypothesis: Trix (triple-smoothed EMA) captures momentum with reduced noise. Weekly trend filter ensures 
alignment with higher timeframe direction, while volume confirmation filters false breakouts. 
Works in bull markets via momentum continuation and in bear via mean-reversion at extremes.
Target: 50-150 trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_11835_6h_trix_weekly_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
TRIX_PERIOD = 12
TRIX_SIGNAL = 9
WEEKLY_EMA_PERIOD = 20
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_trix(close, period):
    """Calculate TRIX: triple EMA of percent change"""
    ema1 = pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean()
    ema2 = ema1.ewm(span=period, adjust=False, min_periods=period).mean()
    ema3 = ema2.ewm(span=period, adjust=False, min_periods=period).mean()
    # Calculate percent change of triple EMA
    pct_change = ema3.pct_change()
    # Signal line is EMA of TRIX
    trix = pct_change * 100  # Scale for readability
    trix_signal = trix.ewm(span=TRIX_SIGNAL, adjust=False, min_periods=TRIX_SIGNAL).mean()
    return trix.values, trix_signal.values

def calculate_ema(close, period):
    """Calculate EMA"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_atr(high, low, close, period):
    """Calculate ATR"""
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
    
    # Load weekly data ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA for trend filter
    ema_weekly = calculate_ema(df_weekly['close'].values, WEEKLY_EMA_PERIOD)
    ema_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_weekly)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    trix, trix_signal = calculate_trix(close, TRIX_PERIOD)
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(TRIX_PERIOD + TRIX_SIGNAL, WEEKLY_EMA_PERIOD, VOLUME_MA_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if weekly EMA not available
        if np.isnan(ema_weekly_aligned[i]):
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
        
        # Trix momentum signals
        trix_cross_up = trix[i] > trix_signal[i] and trix[i-1] <= trix_signal[i-1] if i > 0 and not np.isnan(trix[i]) and not np.isnan(trix_signal[i]) else False
        trix_cross_down = trix[i] < trix_signal[i] and trix[i-1] >= trix_signal[i-1] if i > 0 and not np.isnan(trix[i]) and not np.isnan(trix_signal[i]) else False
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Weekly trend filter
        weekly_uptrend = close[i] > ema_weekly_aligned[i]
        weekly_downtrend = close[i] < ema_weekly_aligned[i]
        
        # Entry conditions: momentum + volume + trend alignment
        long_entry = trix_cross_up and volume_ok and weekly_uptrend
        short_entry = trix_cross_down and volume_ok and weekly_downtrend
        
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