#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6s strategy using 6h momentum with 12h trend filter and volume confirmation.
# Long when 6h RSI > 50, 12h EMA(20) is rising, and volume > 1.5x average.
# Short when 6h RSI < 50, 12h EMA(20) is falling, and volume > 1.5x average.
# Trend filter prevents counter-trend trades, reducing whipsaw in choppy markets.
# Targets 75-150 total trades over 4 years (19-38/year) to balance opportunity and cost.

name = "exp_13879_6h_12h_ema_rsi_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
RSI_PERIOD = 14
EMA_PERIOD = 20
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
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
    
    # Load 12h data for EMA trend filter ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h EMA for trend direction
    close_12h = df_12h['close'].values
    ema_12h = calculate_ema(close_12h, EMA_PERIOD)
    ema_12h_rising = ema_12h > np.roll(ema_12h, 1)  # Current EMA > previous EMA
    ema_12h_rising[0] = False  # First value has no previous
    ema_12h_falling = ema_12h < np.roll(ema_12h, 1)  # Current EMA < previous EMA
    ema_12h_falling[0] = False
    
    # Align 12h EMA trend to 6h timeframe
    ema_rising_aligned = align_htf_to_ltf(prices, df_12h, ema_12h_rising)
    ema_falling_aligned = align_htf_to_ltf(prices, df_12h, ema_12h_falling)
    
    # 6h data for RSI, ATR, and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # RSI for momentum filter
    rsi = calculate_rsi(close, RSI_PERIOD)
    
    # ATR for stop loss
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Volume confirmation
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(RSI_PERIOD, EMA_PERIOD, VOLUME_MA_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(rsi[i]) or np.isnan(atr[i]) or np.isnan(volume_ma[i]) or \
           np.isnan(ema_rising_aligned[i]) or np.isnan(ema_falling_aligned[i]):
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
        
        # Momentum filter from RSI
        rsi_bullish = rsi[i] > 50
        rsi_bearish = rsi[i] < 50
        
        # Trend filter from 12h EMA
        trend_up = ema_rising_aligned[i]
        trend_down = ema_falling_aligned[i]
        
        # Entry signals
        long_signal = volume_ok and rsi_bullish and trend_up
        short_signal = volume_ok and rsi_bearish and trend_down
        
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
            # Exit long when trend turns down or RSI < 40
            if not trend_up or rsi[i] < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            # Exit short when trend turns up or RSI > 60
            if not trend_down or rsi[i] > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals