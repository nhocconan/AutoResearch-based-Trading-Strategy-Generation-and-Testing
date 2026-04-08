# 1h MACD with 4h Trend Filter and Volume Confirmation
# Hypothesis: Use MACD on 1h for entry timing with 4h EMA for trend direction and volume confirmation.
# This filters out counter-trend trades and ensures entries have momentum.
# Target: 60-150 total trades over 4 years (15-37/year) to minimize fee drag.
# Works in both bull and bear markets by following the 4h trend.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_13894_1h_macd4h_trend_vol_v1"
timeframe = "1h"
leverage = 1.0

# Parameters
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
EMA_TREND = 50
VOLUME_MA = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.20
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_macd(close, fast, slow, signal):
    """Calculate MACD line, signal line, and histogram"""
    ema_fast = pd.Series(close).ewm(span=fast, adjust=False, min_periods=fast).mean().values
    ema_slow = pd.Series(close).ewm(span=slow, adjust=False, min_periods=slow).mean().values
    macd_line = ema_fast - ema_slow
    signal_line = pd.Series(macd_line).ewm(span=signal, adjust=False, min_periods=signal).mean().values
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def calculate_ema(close, period):
    """Calculate EMA"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 4h data for trend filter ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h EMA for trend direction
    close_4h = df_4h['close'].values
    ema_4h = calculate_ema(close_4h, EMA_TREND)
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # 1h data for MACD, ATR, and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # MACD for momentum
    macd_line, signal_line, _ = calculate_macd(close, MACD_FAST, MACD_SLOW, MACD_SIGNAL)
    
    # ATR for stop loss
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Volume confirmation
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA, min_periods=VOLUME_MA).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(MACD_SLOW, MACD_SIGNAL, EMA_TREND, VOLUME_MA) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(ema_4h_aligned[i]) or np.isnan(macd_line[i]) or np.isnan(signal_line[i]) or np.isnan(volume_ma[i]):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        # Check stops
        if position == 1:  # long position
            # Check stop loss
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # short position
            # Check stop loss
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD)
        
        # Trend filter from 4h EMA
        trend_up = close[i] > ema_4h_aligned[i]
        trend_down = close[i] < ema_4h_aligned[i]
        
        # MACD signals
        macd_bullish = macd_line[i] > signal_line[i]
        macd_bearish = macd_line[i] < signal_line[i]
        
        # Entry signals
        long_signal = volume_ok and trend_up and macd_bullish
        short_signal = volume_ok and trend_down and macd_bearish
        
        # Generate signals
        if position == 0:
            if long_signal:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif short_signal:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long on MACD bearish cross
            if macd_line[i] < signal_line[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            # Exit short on MACD bullish cross
            if macd_line[i] > signal_line[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals