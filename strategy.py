#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_12744_1d_kama_rsi_chop_v1"
timeframe = "1d"
leverage = 1.0

# Parameters
KAMA_EFFICIENCY_PERIOD = 10
KAMA_FAST_EMA = 2
KAMA_SLOW_EMA = 30
RSI_PERIOD = 14
RSI_OVERBOUGHT = 65
RSI_OVERSOLD = 35
CHOPPINESS_PERIOD = 14
CHOPPINESS_THRESHOLD = 61.8
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def calculate_kama(close, eff_period, fast, slow):
    """Calculate Kaufman Adaptive Moving Average"""
    change = np.abs(close - np.roll(close, eff_period))
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0]))[eff_period:], axis=0) if eff_period > 0 else np.zeros_like(close)
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1))**2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    return kama

def calculate_rsi(close, period):
    """Calculate Relative Strength Index"""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_choppiness(high, low, close, period):
    """Calculate Choppiness Index"""
    atr = calculate_atr(high, low, close, 1)
    sum_atr = pd.Series(atr).rolling(window=period, min_periods=period).sum().values
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    range_hl = highest_high - lowest_low
    cpi = 100 * np.log10(sum_atr / range_hl) / np.log10(period)
    return cpi

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
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly KAMA for trend direction
    close_1w = df_1w['close'].values
    kama_1w = calculate_kama(close_1w, KAMA_EFFICIENCY_PERIOD, KAMA_FAST_EMA, KAMA_SLOW_EMA)
    kama_1w_prev = np.roll(kama_1w, 1)
    kama_1w_prev[0] = kama_1w[0]
    kama_trend = kama_1w > kama_1w_prev  # 1 for up, 0 for down
    
    # Align weekly KAMA trend to daily timeframe
    kama_trend_aligned = align_htf_to_ltf(prices, df_1w, kama_trend.astype(float))
    
    # Calculate daily indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    rsi = calculate_rsi(close, RSI_PERIOD)
    chop = calculate_choppiness(high, low, close, CHOPPINESS_PERIOD)
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(RSI_PERIOD, CHOPPINESS_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if weekly trend not available
        if np.isnan(kama_trend_aligned[i]):
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
        
        # Conditions
        kama_up = kama_trend_aligned[i] > 0.5
        rsi_not_overbought = rsi[i] < RSI_OVERBOUGHT
        rsi_not_oversold = rsi[i] > RSI_OVERSOLD
        chop_high = chop[i] > CHOPPINESS_THRESHOLD  # choppy market
        
        # Long: KAMA up + RSI not overbought + choppy (mean reversion in chop)
        long_entry = kama_up and rsi_not_overbought and chop_high
        # Short: KAMA down + RSI not oversold + choppy (mean reversion in chop)
        short_entry = not kama_up and rsi_not_oversold and chop_high
        
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