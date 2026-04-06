#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h trend filter with 1h pullback entry and volume confirmation.
# In both bull and bear markets, trends exist but require pullbacks for optimal entry.
# 4h EMA determines trend direction, 1h RSI identifies pullbacks, volume confirms momentum.
# Target: 80-150 trades over 4 years (20-38/year) to balance opportunity and cost.
# Session filter (08-20 UTC) reduces noise during low-liquidity hours.

name = "exp_12914_1h_4h_trend_pullback_v1"
timeframe = "1h"
leverage = 1.0

# Parameters
EMA_FAST = 9
EMA_SLOW = 21
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.20
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def calculate_rsi(close, period):
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def calculate_atr(high, low, close, period):
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
    
    # Calculate 4h EMA for trend
    close_4h = df_4h['close'].values
    ema_fast = pd.Series(close_4h).ewm(span=EMA_FAST, adjust=False, min_periods=EMA_FAST).mean().values
    ema_slow = pd.Series(close_4h).ewm(span=EMA_SLOW, adjust=False, min_periods=EMA_SLOW).mean().values
    trend_up = ema_fast > ema_slow
    
    # Align trend to 1h
    trend_up_aligned = align_htf_to_ltf(prices, df_4h, trend_up)
    
    # Calculate 1h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    rsi = calculate_rsi(close, RSI_PERIOD)
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Pre-compute session hours
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    start = max(EMA_SLOW, RSI_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Session filter: 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        # Skip if trend data not available
        if np.isnan(trend_up_aligned[i]):
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
        
        # Entry logic: pullback in trend direction
        if trend_up_aligned[i]:  # uptrend
            # Look for RSI pullback from overbought
            if rsi[i] < RSI_OVERBOUGHT and rsi[i-1] >= RSI_OVERBOUGHT:
                if volume_ok:
                    signals[i] = SIGNAL_SIZE
                    position = 1
                    entry_price = close[i]
                    stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
                else:
                    signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            else:
                signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
        else:  # downtrend
            # Look for RSI bounce from oversold
            if rsi[i] > RSI_OVERSOLD and rsi[i-1] <= RSI_OVERSOLD:
                if volume_ok:
                    signals[i] = -SIGNAL_SIZE
                    position = -1
                    entry_price = close[i]
                    stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
                else:
                    signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            else:
                signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
    
    return signals