#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-hour RSI(2) extreme mean reversion with 4-hour trend filter and volume confirmation.
# In bull markets, buy RSI(2) < 5 during uptrend; in bear markets, sell RSI(2) > 95 during downtrend.
# The 4-hour RSI(14) determines trend direction to avoid counter-trend trades.
# Volume > 1.5x average confirms genuine momentum exhaustion.
# Target: 100-200 total trades over 4 years (25-50/year) balanced for 1h frequency.

name = "exp_13194_1h_rsi2_extreme_4h_rsi14_vol_v1"
timeframe = "1h"
leverage = 1.0

# Parameters
RSI2_PERIOD = 2
RSI14_PERIOD = 14
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
RSI2_LONG_THRESHOLD = 5
RSI2_SHORT_THRESHOLD = 95
SIGNAL_SIZE = 0.20
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_rsi(close, period):
    """Calculate RSI using Wilder's smoothing"""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi

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
    
    # Load 4h data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h RSI for trend filter
    close_4h = df_4h['close'].values
    rsi_14_4h = calculate_rsi(close_4h, RSI14_PERIOD)
    rsi_14_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_14_4h)
    
    # Calculate 1h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # RSI(2)
    rsi_2 = calculate_rsi(close, RSI2_PERIOD)
    
    # Volume MA
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(RSI2_PERIOD, RSI14_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if 4h RSI not available
        if np.isnan(rsi_14_4h_aligned[i]):
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
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Trend filter from 4h RSI
        uptrend_4h = rsi_14_4h_aligned[i] > 50
        downtrend_4h = rsi_14_4h_aligned[i] < 50
        
        # RSI(2) extreme signals
        rsi2_oversold = rsi_2[i] < RSI2_LONG_THRESHOLD
        rsi2_overbought = rsi_2[i] > RSI2_SHORT_THRESHOLD
        
        # Generate signals
        if position == 0:
            if rsi2_oversold and uptrend_4h and volume_ok:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif rsi2_overbought and downtrend_4h and volume_ok:
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