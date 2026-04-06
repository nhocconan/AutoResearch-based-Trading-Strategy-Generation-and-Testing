#!/usr/bin/env python3
"""
12h KAMA + RSI + Chop Filter
Hypothesis: KAMA adapts to market noise, capturing true trends while filtering whipsaws.
RSI filters overextended entries. Chop regime filter ensures we trade only in trending markets.
Works in bull via trend continuation, bear via trend reversals with chop filter avoiding false signals.
Target: 75-250 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_12445_12h_kama_rsi_chop_v1"
timeframe = "12h"
leverage = 1.0

# Parameters
KAMA_ER_FAST = 2
KAMA_ER_SLOW = 30
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
CHOP_PERIOD = 14
CHOP_THRESHOLD = 61.8  # >61.8 = ranging, <38.2 = trending
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5
SIGNAL_SIZE = 0.25

def calculate_kama(close, er_fast, er_slow):
    """Calculate Kaufman Adaptive Moving Average"""
    close_series = pd.Series(close)
    change = abs(close_series - close_series.shift(er_slow))
    volatility = abs(close_series.diff()).rolling(window=er_slow).sum()
    er = change / volatility
    er = er.fillna(0)
    sc = (er * (2/(er_fast+1) - 2/(er_slow+1)) + 2/(er_slow+1)) ** 2
    kama = [close_series.iloc[0]]  # seed
    for i in range(1, len(close_series)):
        kama.append(kama[-1] + sc.iloc[i] * (close_series.iloc[i] - kama[-1]))
    return np.array(kama)

def calculate_rsi(close, period):
    """Calculate Relative Strength Index"""
    delta = pd.Series(close).diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50).values

def calculate_chop(high, low, close, period):
    """Calculate Choppiness Index"""
    atr = []
    tr1 = pd.Series(high) - pd.Series(low)
    tr2 = abs(pd.Series(high) - pd.Series(close).shift(1))
    tr3 = abs(pd.Series(low) - pd.Series(close).shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_sum = tr.rolling(window=period, min_periods=period).sum()
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max()
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min()
    chop = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(period)
    return chop.fillna(50).values

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
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily EMA for trend filter
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate 12h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    kama = calculate_kama(close, KAMA_ER_FAST, KAMA_ER_SLOW)
    rsi = calculate_rsi(close, RSI_PERIOD)
    chop = calculate_chop(high, low, close, CHOP_PERIOD)
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(50, KAMA_ER_SLOW, RSI_PERIOD, CHOP_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if daily EMA not available
        if np.isnan(ema_1d_aligned[i]):
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
        
        # Regime filter: only trade when NOT chopping (trending market)
        trending_market = chop[i] < CHOP_THRESHOLD  # chop < 61.8 = trending
        
        # KAMA trend
        price_above_kama = close[i] > kama[i]
        price_below_kama = close[i] < kama[i]
        
        # RSI filter (avoid extremes)
        rsi_not_overbought = rsi[i] < RSI_OVERBOUGHT
        rsi_not_oversold = rsi[i] > RSI_OVERSOLD
        
        # Daily trend filter
        uptrend_1d = close[i] > ema_1d_aligned[i]
        downtrend_1d = close[i] < ema_1d_aligned[i]
        
        # Entry conditions
        long_entry = trending_market and price_above_kama and rsi_not_overbought and uptrend_1d
        short_entry = trending_market and price_below_kama and rsi_not_oversold and downtrend_1d
        
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