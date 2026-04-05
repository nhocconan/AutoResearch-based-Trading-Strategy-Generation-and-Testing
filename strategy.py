#!/usr/bin/env python3
"""
Experiment #10874: 1h Momentum with 4h/1d Trend Filter and Session Filter
Hypothesis: 1-hour momentum entries (price > VWAP + momentum) in the direction of 4h EMA trend and 1d trend,
with volume confirmation, provide high-probability trades. Session filter (08-20 UTC) reduces noise.
Works in bull markets (trend following) and bear markets (mean reversion within trend).
Target: 60-150 total trades over 4 years (15-37/year) on 1h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_10874_1h_momentum_4h_1d_trend_session_v1"
timeframe = "1h"
leverage = 1.0

# Parameters
RSI_PERIOD = 14
MOMENTUM_PERIOD = 10
VWAP_WINDOW = 20
EMA_4H_PERIOD = 21
EMA_1D_PERIOD = 50
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.20
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_rsi(close, period):
    """Calculate RSI"""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_momentum(close, period):
    """Calculate momentum (rate of change)"""
    mom = np.full_like(close, np.nan)
    mom[period:] = (close[period:] - close[:-period]) / close[:-period] * 100
    return mom

def calculate_vwap(high, low, close, volume, window):
    """Calculate VWAP"""
    typical_price = (high + low + close) / 3
    vwap = np.full_like(close, np.nan)
    for i in range(window-1, len(close)):
        tp_slice = typical_price[i-window+1:i+1]
        vol_slice = volume[i-window+1:i+1]
        if vol_slice.sum() > 0:
            vwap[i] = np.dot(tp_slice, vol_slice) / vol_slice.sum()
    return vwap

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
    
    # Load 4h and 1d data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h EMA for trend
    ema_4h = calculate_ema(df_4h['close'].values, EMA_4H_PERIOD)
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Calculate 1d EMA for trend
    ema_1d = calculate_ema(df_1d['close'].values, EMA_1D_PERIOD)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate 1h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    rsi = calculate_rsi(close, RSI_PERIOD)
    momentum = calculate_momentum(close, MOMENTUM_PERIOD)
    vwap = calculate_vwap(high, low, close, volume, VWAP_WINDOW)
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(RSI_PERIOD, MOMENTUM_PERIOD, VWAP_WINDOW, EMA_4H_PERIOD, EMA_1D_PERIOD, VOLUME_MA_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if outside session
        if not session_filter[i]:
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
            
        # Skip if 4h or 1d EMA not available
        if np.isnan(ema_4h_aligned[i]) or np.isnan(ema_1d_aligned[i]):
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
        
        # Momentum and VWAP conditions
        price_above_vwap = close[i] > vwap[i] if not np.isnan(vwap[i]) else False
        momentum_up = momentum[i] > 0 if not np.isnan(momentum[i]) else False
        momentum_down = momentum[i] < 0 if not np.isnan(momentum[i]) else False
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Trend filters
        uptrend_4h = close[i] > ema_4h_aligned[i]
        uptrend_1d = close[i] > ema_1d_aligned[i]
        downtrend_4h = close[i] < ema_4h_aligned[i]
        downtrend_1d = close[i] < ema_1d_aligned[i]
        
        # Entry conditions
        long_entry = (price_above_vwap and momentum_up and volume_ok and 
                     uptrend_4h and uptrend_1d)
        short_entry = ((not price_above_vwap) and momentum_down and volume_ok and
                      downtrend_4h and downtrend_1d)
        
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