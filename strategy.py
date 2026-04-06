#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily EMA crossover with weekly trend filter and volume confirmation
# Works in bull/bear: EMA crossover captures momentum, weekly EMA filter ensures
# alignment with higher timeframe trend, volume confirmation filters weak signals.
# Target: 50-150 trades over 4 years (12-38/year) to balance opportunity and cost.

name = "ema_crossover_weekly_trend_v1"
timeframe = "1d"
leverage = 1.0

# Parameters
FAST_EMA = 9
SLOW_EMA = 21
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def calculate_ema(prices, period):
    """Calculate EMA with proper Wilder's smoothing"""
    return pd.Series(prices).ewm(alpha=2/(period+1), adjust=False, min_periods=period).mean().values

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
    
    # Load weekly data ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA for trend filter
    weekly_close = df_weekly['close'].values
    weekly_ema = calculate_ema(weekly_close, 21)
    weekly_ema_aligned = align_htf_to_ltf(prices, df_weekly, weekly_ema)
    
    # Calculate daily indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    fast_ema = calculate_ema(close, FAST_EMA)
    slow_ema = calculate_ema(close, SLOW_EMA)
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(FAST_EMA, SLOW_EMA, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if weekly EMA not available
        if np.isnan(weekly_ema_aligned[i]):
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
        
        # EMA crossover signals
        ema_bullish = fast_ema[i] > slow_ema[i]
        ema_bearish = fast_ema[i] < slow_ema[i]
        
        # Weekly trend filter: only trade in direction of weekly trend
        weekly_bullish = close[i] > weekly_ema_aligned[i]
        weekly_bearish = close[i] < weekly_ema_aligned[i]
        
        # Generate signals
        if position == 0:
            # Long: bullish EMA crossover + above weekly EMA + volume
            if ema_bullish and weekly_bullish and volume_ok:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            # Short: bearish EMA crossover + below weekly EMA + volume
            elif ema_bearish and weekly_bearish and volume_ok:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Maintain long position
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            # Maintain short position
            signals[i] = -SIGNAL_SIZE
    
    return signals