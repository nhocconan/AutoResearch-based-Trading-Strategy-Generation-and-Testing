#!/usr/bin/env python3
"""
Experiment #8894: 1h strategy with 4h/1d trend filter + volume confirmation.
Hypothesis: Use 4h/1d trends for direction and 1h for precise entry to reduce whipsaw.
Volume confirms institutional participation. Targets 60-150 trades over 4 years (15-37/year).
Uses session filter (08-20 UTC) to avoid low-volume periods. Works in bull via breakouts and bear via filtered shorts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_8894_1h_4h_1d_trend_vol_sess_v1"
timeframe = "1h"
leverage = 1.0

# Parameters
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0
SIGNAL_SIZE = 0.20
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
TREND_EMA_PERIOD = 34

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
    ema_4h = pd.Series(close_4h).ewm(span=TREND_EMA_PERIOD, adjust=False, min_periods=TREND_EMA_PERIOD).mean().values
    trend_4h = np.where(close_4h > ema_4h, 1, -1)  # 1=uptrend, -1=downtrend
    trend_4h_aligned = align_htf_to_ltf(prices, df_4h, trend_4h)
    
    # Calculate 1d EMA for stronger trend filter
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=TREND_EMA_PERIOD, adjust=False, min_periods=TREND_EMA_PERIOD).mean().values
    trend_1d = np.where(close_1d > ema_1d, 1, -1)  # 1=uptrend, -1=downtrend
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # LTF indicators (1h)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume moving average
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(TREND_EMA_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if outside trading session or HTF data not available
        if not in_session[i] or np.isnan(trend_4h_aligned[i]) or np.isnan(trend_1d_aligned[i]):
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
        
        # Determine market bias: require both 4h and 1d to agree
        bull_bias = (trend_4h_aligned[i] == 1) and (trend_1d_aligned[i] == 1)
        bear_bias = (trend_4h_aligned[i] == -1) and (trend_1d_aligned[i] == -1)
        
        # Volume confirmation
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Entry conditions: only trade with trend and volume
        long_entry = bull_bias and volume_confirmed
        short_entry = bear_bias and volume_confirmed
        
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