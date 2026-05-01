#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA direction + RSI(14) + 1w chop regime filter.
# Long when KAMA is rising (bullish trend), RSI < 30 (oversold), and weekly chop > 61.8 (range market).
# Short when KAMA is falling (bearish trend), RSI > 70 (overbought), and weekly chop > 61.8.
# Uses discrete sizing 0.25. ATR(14) stoploss: signal→0 when price moves against position by 2.5*ATR.
# Session filter: 08-20 UTC to reduce noise. Target: 15-25 trades/year to minimize fee drag.
# Works in bull markets via trend following (KAMA direction) and in bear markets via mean reversion (RSI extremes) in range regimes (chop filter).

name = "1d_KAMA_RSI_1wChop_Regime_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours for 08-20 UTC filter
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # Calculate ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr0 = np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])
    tr = np.concatenate([[tr0], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate KAMA(10, 2, 30) - ER = 2, Fastest EMA = 2, Slowest EMA = 30
    def kama(close, er_length=10, fast=2, slow=30):
        change = np.abs(np.diff(close, n=er_length))
        volatility = np.sum(np.abs(np.diff(close)), axis=0)
        er = np.where(volatility != 0, change / volatility, 0)
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        kama = np.zeros_like(close)
        kama[0] = close[0]
        for i in range(1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    kama_vals = kama(close)
    kama_rising = kama_vals > np.roll(kama_vals, 1)
    kama_falling = kama_vals < np.roll(kama_vals, 1)
    
    # Calculate RSI(14)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.concatenate([[50], rsi])  # pad first value
    
    # Load 1w data ONCE before loop for chop regime (HTF filter)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # Calculate 1w Chop Index(14) - same formula as before
    def true_range(h, l, c):
        tr1 = h[1:] - l[1:]
        tr2 = np.abs(h[1:] - np.roll(c, 1)[1:])
        tr3 = np.abs(l[1:] - np.roll(c, 1)[1:])
        tr0 = np.max([h[0] - l[0], np.abs(h[0] - c[0]), np.abs(l[0] - c[0])])
        tr = np.concatenate([[tr0], np.maximum(tr1, np.maximum(tr2, tr3))])
        atr_ = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
        highest_high = pd.Series(h).rolling(window=14, min_periods=14).max().values
        lowest_low = pd.Series(l).rolling(window=14, min_periods=14).min().values
        chop = 100 * np.log10(atr_ / (highest_high - lowest_low)) / np.log10(14)
        return chop
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    chop_1w = true_range(high_1w, low_1w, close_1w)
    
    # Align 1w chop to 1d
    chop_1w_aligned = align_htf_to_ltf(prices, df_1w, chop_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # track entry price for stoploss
    
    start_idx = 30  # warmup for KAMA and RSI
    
    for i in range(start_idx, n):
        # Session filter: 08-20 UTC
        if not (8 <= hours[i] <= 20):
            signals[i] = 0.0
            continue
        
        if (np.isnan(atr[i]) or np.isnan(kama_rising[i]) or np.isnan(kama_falling[i]) or 
            np.isnan(rsi[i]) or np.isnan(chop_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        
        # Chop regime filter: only trade in range market (chop > 61.8)
        chop_filter = chop_1w_aligned[i] > 61.8
        
        if position == 0:  # Flat - look for new entries
            # Long: KAMA rising AND RSI < 30 (oversold) AND chop regime
            if (kama_rising[i] and 
                rsi[i] < 30 and 
                chop_filter):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: KAMA falling AND RSI > 70 (overbought) AND chop regime
            elif (kama_falling[i] and 
                  rsi[i] > 70 and 
                  chop_filter):
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Stoploss: price moves against position by 2.5*ATR
            if curr_close < entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: KAMA turns bearish OR chop regime ends (trending)
            elif (not kama_rising[i]) or \
                 chop_1w_aligned[i] <= 61.8:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: price moves against position by 2.5*ATR
            if curr_close > entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: KAMA turns bullish OR chop regime ends (trending)
            elif (not kama_falling[i]) or \
                 chop_1w_aligned[i] <= 61.8:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
    
    return signals