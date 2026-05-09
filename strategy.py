#!/usr/bin/env python3
# Hypothesis: 1d KAMA direction + RSI + Chop regime filter (1d) with 1w trend filter
# Uses 1d KAMA as primary trend filter, RSI(14) for overbought/oversold entries, and Choppiness Index to avoid choppy markets
# Adds 1w EMA(34) trend filter to only trade in direction of weekly trend
# Long when: KAMA rising, RSI < 30, CHOP > 61.8 (ranging), weekly EMA rising
# Short when: KAMA falling, RSI > 70, CHOP > 61.8 (ranging), weekly EMA falling
# Exit when: RSI crosses 50 (mean reversion) OR trend changes
# Position size: 0.25 to limit drawdown. Target: 10-25 trades/year.

name = "1d_KAMA_RSI_Chop_Regime_1wTrend"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_kama(close, er_period=10, fast_sc=2, slow_sc=30):
    """Kaufman Adaptive Moving Average"""
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close)), axis=0) if len(change.shape) == 0 else pd.Series(change).rolling(er_period, min_periods=1).sum().values
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/(fast_sc+1) - 2/(slow_sc+1)) + 2/(slow_sc+1)) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    return kama

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index: higher = ranging, lower = trending"""
    atr = np.abs(high - low)
    tr1 = np.abs(high - np.roll(close, 1))
    tr2 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(atr, np.maximum(tr1, tr2))
    tr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum()
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max()
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min()
    range_ = highest_high - lowest_low
    chop = 100 * np.log10(tr_sum / range_) / np.log10(period)
    return chop.values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d KAMA for trend
    kama = calculate_kama(close, er_period=10, fast_sc=2, slow_sc=30)
    kama_prev = np.roll(kama, 1)
    kama_prev[0] = kama[0]
    kama_rising = kama > kama_prev
    kama_falling = kama < kama_prev
    
    # 1d RSI for entry
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # 1d Choppiness Index for regime
    chop = calculate_choppiness(high, low, close, period=14)
    chop_high = chop > 61.8  # ranging market
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Weekly EMA(34) for trend filter
    close_1w = df_1w['close']
    ema_34_1w = close_1w.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_prev = np.roll(ema_34_1w, 1)
    ema_34_1w_prev[0] = ema_34_1w[0]
    ema_rising_1w = ema_34_1w > ema_34_1w_prev
    ema_falling_1w = ema_34_1w < ema_34_1w_prev
    ema_rising_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_rising_1w)
    ema_falling_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_falling_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(kama_rising[i]) or np.isnan(kama_falling[i]) or
            np.isnan(rsi[i]) or np.isnan(chop_high[i]) or
            np.isnan(ema_rising_1w_aligned[i]) or np.isnan(ema_falling_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: KAMA rising, RSI oversold, choppy market, weekly trend up
            if (kama_rising[i] and 
                rsi[i] < 30 and 
                chop_high[i] and 
                ema_rising_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: KAMA falling, RSI overbought, choppy market, weekly trend down
            elif (kama_falling[i] and 
                  rsi[i] > 70 and 
                  chop_high[i] and 
                  ema_falling_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: RSI crosses 50 OR weekly trend changes
            if (rsi[i] > 50) or (not ema_rising_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RSI crosses 50 OR weekly trend changes
            if (rsi[i] < 50) or (not ema_falling_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals