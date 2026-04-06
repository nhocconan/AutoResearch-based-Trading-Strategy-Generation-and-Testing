#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-day KAMA trend with RSI(14) filter and weekly volatility regime filter
# Works in bull/bear because KAMA adapts to market noise, RSI filters extremes,
# and weekly volatility regime identifies trending vs ranging markets.
# Target: 50-100 trades over 4 years (12-25/year) to balance opportunity and cost.

name = "exp_12918_1d_kama_rsi_vol_regime_v1"
timeframe = "1d"
leverage = 1.0

# Parameters
KAMA_ER_FAST = 2
KAMA_ER_SLOW = 30
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
VOLATILITY_PERIOD = 20
VOLATILITY_THRESHOLD = 0.5  # ATR ratio threshold for trending regime
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def calculate_kama(close, er_fast, er_slow):
    """Calculate Kaufman Adaptive Moving Average"""
    change = np.abs(close - np.roll(close, 10))
    volatility = np.sum(np.abs(np.diff(close, n=1)), axis=0)
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/(er_fast+1) - 2/(er_slow+1)) + 2/(er_slow+1)) ** 2
    kama = np.copy(close)
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    return kama

def calculate_rsi(close, period):
    """Calculate Relative Strength Index"""
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

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
    
    # Calculate weekly ATR for volatility regime
    high_w = df_weekly['high'].values
    low_w = df_weekly['low'].values
    close_w = df_weekly['close'].values
    atr_w = calculate_atr(high_w, low_w, close_w, ATR_PERIOD)
    
    # Calculate weekly volatility ratio (current ATR / average ATR)
    atr_ma_w = pd.Series(atr_w).rolling(window=VOLATILITY_PERIOD, min_periods=VOLATILITY_PERIOD).mean().values
    volatility_ratio = atr_w / atr_ma_w
    volatility_ratio = np.where(atr_ma_w > 0, volatility_ratio, 1.0)
    
    # Align volatility ratio to daily timeframe
    volatility_ratio_aligned = align_htf_to_ltf(prices, df_weekly, volatility_ratio)
    
    # Calculate daily indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    kama = calculate_kama(close, KAMA_ER_FAST, KAMA_ER_SLOW)
    rsi = calculate_rsi(close, RSI_PERIOD)
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(KAMA_ER_SLOW, RSI_PERIOD, ATR_PERIOD, VOLATILITY_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if volatility ratio not available
        if np.isnan(volatility_ratio_aligned[i]):
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
        
        # Trend regime filter: only trade when volatility ratio > threshold (trending market)
        is_trending = volatility_ratio_aligned[i] > VOLATILITY_THRESHOLD
        
        # KAMA direction: price above KAMA = uptrend, below = downtrend
        price_above_kama = close[i] > kama[i]
        price_below_kama = close[i] < kama[i]
        
        # RSI filter: avoid overbought/oversold extremes
        rsi_not_overbought = rsi[i] < RSI_OVERBOUGHT
        rsi_not_oversold = rsi[i] > RSI_OVERSOLD
        
        # Generate signals
        if position == 0:
            if is_trending and price_above_kama and rsi_not_overbought:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif is_trending and price_below_kama and rsi_not_oversold:
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