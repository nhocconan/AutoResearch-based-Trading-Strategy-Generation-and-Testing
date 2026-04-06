#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_13898_1d_kama_rsi_chop_v1"
timeframe = "1d"
leverage = 1.0

# Parameters
KAMA_PERIOD = 10
FAST_EMA = 2
SLOW_EMA = 30
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
CHOP_PERIOD = 14
CHOP_THRESHOLD = 61.8
SIGNAL_SIZE = 0.30
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def calculate_kama(close, period, fast, slow):
    """Calculate Kaufman's Adaptive Moving Average"""
    close_series = pd.Series(close)
    diff = np.abs(close_series.diff(1)).values
    diff[0] = 0
    change = np.abs(close_series - np.roll(close_series, period))
    change[:period] = 0
    er = np.zeros_like(close)
    er[period:] = change[period:] / (np.convolve(diff, np.ones(period), 'valid') + 1e-10)
    sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    return kama

def calculate_rsi(close, period):
    """Calculate Relative Strength Index"""
    delta = np.diff(close)
    seed = delta[:period]
    up = seed[seed >= 0].sum() / period
    down = -seed[seed < 0].sum() / period
    if down == 0:
        rs = 0
    else:
        rs = up / down
    rsi = np.zeros_like(close)
    rsi[:period] = 100. - (100. / (1. + rs))
    for i in range(period, len(close)):
        delta = close[i] - close[i-1]
        if delta > 0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta
        up = (up * (period-1) + upval) / period
        down = (down * (period-1) + downval) / period
        if down == 0:
            rs = 0
        else:
            rs = up / down
        rsi[i] = 100. - (100. / (1. + rs))
    return rsi

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    atr = np.zeros_like(close)
    atr[0] = tr[0]
    for i in range(1, len(close)):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    return atr

def calculate_chop(high, low, close, period):
    """Calculate Choppiness Index"""
    atr = calculate_atr(high, low, close, 1)
    sum_atr = np.zeros_like(close)
    for i in range(len(close)):
        if i < period:
            sum_atr[i] = np.sum(atr[:i+1])
        else:
            sum_atr[i] = np.sum(atr[i-period+1:i+1])
    max_high = np.zeros_like(close)
    min_low = np.zeros_like(close)
    for i in range(len(close)):
        if i < period:
            max_high[i] = np.max(high[:i+1])
            min_low[i] = np.min(low[:i+1])
        else:
            max_high[i] = np.max(high[i-period+1:i+1])
            min_low[i] = np.min(low[i-period+1:i+1])
    chop = np.zeros_like(close)
    for i in range(len(close)):
        if max_high[i] != min_low[i] and sum_atr[i] > 0:
            chop[i] = 100 * np.log10(sum_atr[i] / (max_high[i] - min_low[i])) / np.log10(period)
        else:
            chop[i] = 50
    return chop

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data for regime filter ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA200 for trend regime
    close_weekly = df_weekly['close'].values
    ema200_weekly = pd.Series(close_weekly).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema200_weekly)
    
    # Daily data for indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # KAMA for trend direction
    kama = calculate_kama(close, KAMA_PERIOD, FAST_EMA, SLOW_EMA)
    
    # RSI for momentum
    rsi = calculate_rsi(close, RSI_PERIOD)
    
    # Choppiness Index for regime
    chop = calculate_chop(high, low, close, CHOP_PERIOD)
    
    # ATR for stop loss
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(KAMA_PERIOD, RSI_PERIOD, CHOP_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or np.isnan(ema200_weekly_aligned[i]):
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
        
        # Regime filter: only trade in ranging markets (chop > threshold)
        market_ranging = chop[i] > CHOP_THRESHOLD
        
        # KAMA trend direction
        price_above_kama = close[i] > kama[i]
        price_below_kama = close[i] < kama[i]
        
        # RSI momentum
        rsi_oversold = rsi[i] < RSI_OVERSOLD
        rsi_overbought = rsi[i] > RSI_OVERBOUGHT
        
        # Entry signals
        long_signal = market_ranging and price_above_kama and rsi_oversold
        short_signal = market_ranging and price_below_kama and rsi_overbought
        
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
            # Exit long on RSI overbought or price below KAMA
            if rsi[i] > RSI_OVERBOUGHT or close[i] < kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            # Exit short on RSI oversold or price above KAMA
            if rsi[i] < RSI_OVERSOLD or close[i] > kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals