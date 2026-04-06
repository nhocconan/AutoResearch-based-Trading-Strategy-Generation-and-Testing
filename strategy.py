#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_12804_1d_1w_kama_rsi_chop_v1"
timeframe = "1d"
leverage = 1.0

# Parameters
KAMA_ER_FAST = 2
KAMA_ER_SLOW = 30
RSI_PERIOD = 14
RSI_OVERBOUGHT = 65
RSI_OVERSOLD = 35
CHOPPINESS_PERIOD = 14
CHOPPINESS_THRESHOLD = 61.8  # >61.8 = choppy (range)
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0
SIGNAL_SIZE = 0.25
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5

def calculate_kama(close, er_fast, er_slow):
    """Calculate Kaufman Adaptive Moving Average"""
    change = np.abs(close - np.roll(close, 10))
    volatility = np.sum(np.abs(np.diff(close, n=1)), axis=0)
    # Handle first element
    volatility = np.concatenate([[np.sum(np.abs(np.diff(close[:11])))], volatility[1:]])
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/(er_fast+1) - 2/(er_slow+1)) + 2/(er_slow+1)) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    return kama

def calculate_rsi(close, period):
    """Calculate RSI"""
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.concatenate([[np.mean(gain[:period])], np.zeros(len(close)-period)])
    avg_loss = np.concatenate([[np.mean(loss[:period])], np.zeros(len(close)-period)])
    for i in range(period, len(close)):
        avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
        avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_choppiness(high, low, close, period):
    """Calculate Choppiness Index"""
    atr = np.zeros_like(close)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    
    # Calculate sum of ATR over period
    atr_sum = pd.Series(atr).rolling(window=period, min_periods=period).sum().values
    # Calculate max(high) - min(low) over period
    max_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    min_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    range_max_min = max_high - min_low
    
    # Avoid division by zero
    choppiness = 100 * np.log10(atr_sum / np.where(range_max_min != 0, range_max_min, 1)) / np.log10(period)
    return choppiness

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
    
    # Calculate weekly KAMA for trend
    close_1w = df_1w['close'].values
    kama_1w = calculate_kama(close_1w, KAMA_ER_FAST, KAMA_ER_SLOW)
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_1w)
    
    # Calculate daily indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    rsi = calculate_rsi(close, RSI_PERIOD)
    chop = calculate_choppiness(high, low, close, CHOPPINESS_PERIOD)
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(KAMA_ER_SLOW, RSI_PERIOD, CHOPPINESS_PERIOD, ATR_PERIOD, VOLUME_MA_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if KAMA not available
        if np.isnan(kama_1w_aligned[i]):
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
        
        # KAMA trend direction
        kama_bullish = close[i] > kama_1w_aligned[i]
        kama_bearish = close[i] < kama_1w_aligned[i]
        
        # RSI conditions
        rsi_overbought = rsi[i] > RSI_OVERBOUGHT
        rsi_oversold = rsi[i] < RSI_OVERSOLD
        
        # Chop filter - only trade in ranging markets
        chop_high = chop[i] > CHOPPINESS_THRESHOLD
        
        # Generate signals
        if position == 0:
            # Long: price above KAMA (uptrend), RSI oversold, choppy market, volume confirmation
            if kama_bullish and rsi_oversold and chop_high and volume_ok:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            # Short: price below KAMA (downtrend), RSI overbought, choppy market, volume confirmation
            elif kama_bearish and rsi_overbought and chop_high and volume_ok:
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