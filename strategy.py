#!/usr/bin/env python3
"""
1d KAMA + RSI + Chop Regime
Hypothesis: KAMA identifies adaptive trend direction, RSI(2) captures short-term momentum extremes,
and Choppiness Index (CHOP) filters regimes: CHOP > 61.8 = range (mean reversion at RSI extremes),
CHOP < 38.2 = trend (follow KAMA direction). Works in bull (follow KAMA up when trending) 
and bear (fade RSI extremes when ranging) via symmetric logic. Target: 15-25 trades/year on 1d
to avoid fee drag and generalize to 2025 bear market.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate KAMA(10, 2, 30) - adaptive trend
    def calculate_kama(close, er_fast=2, er_slow=30):
        change = np.abs(np.diff(close, n=10))
        volatility = np.sum(np.abs(np.diff(close)), axis=0)
        er = np.where(volatility != 0, change / volatility, 0)
        sc = (er * (2/(er_fast+1) - 2/(er_slow+1)) + 2/(er_slow+1)) ** 2
        kama = np.full_like(close, np.nan)
        kama[9] = close[9]  # seed
        for i in range(10, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    # Calculate RSI(2) - short-term momentum
    def calculate_rsi(close, period=2):
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.full_like(close, np.nan)
        avg_loss = np.full_like(close, np.nan)
        avg_gain[period] = np.mean(gain[:period])
        avg_loss[period] = np.mean(loss[:period])
        for i in range(period+1, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    # Calculate Choppiness Index(14) - regime filter
    def calculate_chop(high, low, close, period=14):
        atr = np.zeros(len(close))
        atr[0] = high[0] - low[0]
        for i in range(1, len(close)):
            atr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        sum_atr = np.zeros(len(close))
        for i in range(period, len(close)):
            sum_atr[i] = np.sum(atr[i-period+1:i+1])
        hh = np.zeros(len(close))
        ll = np.zeros(len(close))
        for i in range(period, len(close)):
            hh[i] = np.max(high[i-period+1:i+1])
            ll[i] = np.min(low[i-period+1:i+1])
        chop = np.full(len(close), np.nan)
        for i in range(period, len(close)):
            if hh[i] != ll[i]:
                chop[i] = 100 * np.log10(sum_atr[i] / (hh[i] - ll[i])) / np.log10(period)
        return chop
    
    # Get 1w data for regime confirmation (optional filter)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Calculate weekly EMA34 for trend filter
    close_1w = pd.Series(df_1w['close'])
    ema_34_1w = close_1w.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate indicators
    kama = calculate_kama(close)
    rsi = calculate_rsi(close, 2)
    chop = calculate_chop(high, low, close)
    
    # Align weekly EMA34
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for all indicators
    start_idx = max(30, 2, 14, 34)  # KAMA(30), RSI(2), CHOP(14), EMA34(1w)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or 
            np.isnan(ema_34_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        kama_val = kama[i]
        rsi_val = rsi[i]
        chop_val = chop[i]
        ema_34_1w_val = ema_34_1w_aligned[i]
        
        # Regime definition
        is_ranging = chop_val > 61.8
        is_trending = chop_val < 38.2
        is_transitional = 38.2 <= chop_val <= 61.8  # no clear regime, stay flat
        
        if position == 0:
            # Look for entry signals
            if is_ranging:
                # In ranging market: mean reversion at RSI extremes
                long_entry = rsi_val < 15  # deeply oversold
                short_entry = rsi_val > 85  # deeply overbought
            elif is_trending:
                # In trending market: follow KAMA direction with weekly filter
                long_entry = (curr_close > kama_val) and (curr_close > ema_34_1w_val)
                short_entry = (curr_close < kama_val) and (curr_close < ema_34_1w_val)
            else:
                long_entry = False
                short_entry = False
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit conditions: RSI mean reversion OR trend change
            if is_ranging and rsi_val > 50:  # exit mean reversion at midpoint
                signals[i] = 0.0
                position = 0
            elif not is_trending:  # exit if trend regime ends
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit conditions: RSI mean reversion OR trend change
            if is_ranging and rsi_val < 50:  # exit mean reversion at midpoint
                signals[i] = 0.0
                position = 0
            elif not is_trending:  # exit if trend regime ends
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_KAMA_RSI_ChopRegime"
timeframe = "1d"
leverage = 1.0