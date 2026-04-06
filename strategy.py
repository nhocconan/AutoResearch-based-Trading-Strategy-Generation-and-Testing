#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour CRSI (2) strategy with weekly trend filter and volume confirmation.
# CRSI combines short-term RSI, streak RSI, and percentile rank to identify extreme
# overbought/oversold conditions. In trending markets (weekly EMA), we look for
# pullbacks to extreme CRSI levels. Volume confirms institutional participation.
# Works in bull markets (buy dips in uptrend) and bear markets (sell rallies in downtrend).

name = "exp_13312_12h_crsi_weekly_trend_vol_v1"
timeframe = "12h"
leverage = 1.0

# Parameters
CRSI_PERIOD = 3
STREAK_PERIOD = 2
PERCENT_RANK_LOOKBACK = 100
WEEKLY_EMA_PERIOD = 20
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
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

def calculate_streak_rsi(close, period):
    """Calculate RSI on consecutive up/down days"""
    # Calculate price changes
    changes = np.diff(close, prepend=close[0])
    # Assign +1 for up, -1 for down
    streak = np.where(changes > 0, 1, np.where(changes < 0, -1, 0))
    # Calculate consecutive streaks
    streak_count = np.zeros_like(changes)
    current_streak = 0
    for i in range(len(changes)):
        if streak[i] != 0:
            if i == 0 or streak[i] == streak[i-1]:
                current_streak += streak[i]
            else:
                current_streak = streak[i]
            streak_count[i] = current_streak
        else:
            streak_count[i] = 0
    # Apply RSI to streak counts
    return calculate_rsi(streak_count.astype(float), period)

def calculate_percent_rank(series, lookback):
    """Calculate percentile rank of current value over lookback period"""
    rank = np.full_like(series, np.nan, dtype=float)
    for i in range(len(series)):
        if i < lookback:
            continue
        window = series[i-lookback:i]
        if len(window) == 0:
            continue
        # Calculate percentile rank: percentage of values less than current
        rank[i] = (np.sum(window < series[i]) / len(window)) * 100
    return rank

def calculate_ema(close, period):
    """Calculate EMA"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

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
    if n < 100:
        return np.zeros(n)
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA for trend filter
    close_1w = df_1w['close'].values
    ema_1w = calculate_ema(close_1w, WEEKLY_EMA_PERIOD)
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Calculate 12h indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # CRSI components
    rsi = calculate_rsi(close, CRSI_PERIOD)
    streak_rsi = calculate_streak_rsi(close, STREAK_PERIOD)
    percent_rank = calculate_percent_rank(rsi, PERCENT_RANK_LOOKBACK)
    crsi = (rsi + streak_rsi + percent_rank) / 3.0
    
    # Volume MA
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(WEEKLY_EMA_PERIOD, PERCENT_RANK_LOOKBACK, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if EMA not available
        if np.isnan(ema_1w_aligned[i]) or np.isnan(crsi[i]) or np.isnan(volume_ma[i]) or np.isnan(atr[i]):
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
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD)
        
        # Trend filter: price above/below weekly EMA
        uptrend = close[i] > ema_1w_aligned[i]
        downtrend = close[i] < ema_1w_aligned[i]
        
        # CRSI extreme levels
        crsi_oversold = crsi[i] < 15
        crsi_overbought = crsi[i] > 85
        
        # Generate signals
        if position == 0:
            # Look for pullbacks in trend with volume confirmation
            if uptrend and crsi_oversold and volume_ok:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif downtrend and crsi_overbought and volume_ok:
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