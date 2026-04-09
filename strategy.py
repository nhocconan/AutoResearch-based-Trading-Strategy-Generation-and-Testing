#!/usr/bin/env python3
# 1d_kama_rsi_chop_v1
# Hypothesis: Daily KAMA trend direction with RSI momentum and choppiness regime filter.
# Long when KAMA upward, RSI > 50, and choppy market (CHOP > 61.8) for mean reversion bounce.
# Short when KAMA downward, RSI < 50, and choppy market (CHOP > 61.8) for fade.
# Uses 1w HTF trend filter: only trade in direction of weekly KAMA.
# Discrete sizing (±0.25) to limit fees. Target: 50-100 trades over 4 years.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_kama_rsi_chop_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # KAMA (Kaufman Adaptive Moving Average) - trend indicator
    def kama(close, er_len=10, fast_len=2, slow_len=30):
        close_s = pd.Series(close)
        change = np.abs(close_s - np.roll(close_s, er_len))
        change[0:er_len] = 0
        volatility = np.abs(close_s - close_s.shift(1)).rolling(window=er_len, min_periods=1).sum().values
        volatility[0:er_len] = 1  # avoid division by zero
        er = np.where(volatility > 0, change / volatility, 0)
        sc = (er * (2/(fast_len+1) - 2/(slow_len+1)) + 2/(slow_len+1)) ** 2
        kama = np.zeros_like(close)
        kama[0] = close[0]
        for i in range(1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    # Choppiness Index - regime filter
    def choppiness_index(high, low, close, window=14):
        high_s = pd.Series(high)
        low_s = pd.Series(low)
        close_s = pd.Series(close)
        tr1 = high_s - low_s
        tr2 = np.abs(high_s - close_s.shift(1))
        tr3 = np.abs(low_s - close_s.shift(1))
        tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
        atr = tr.rolling(window=window, min_periods=window).sum()
        hh = high_s.rolling(window=window, min_periods=window).max()
        ll = low_s.rolling(window=window, min_periods=window).min()
        chop = 100 * np.log10(atr / (hh - ll)) / np.log10(window)
        # Handle division by zero and invalid values
        chop = np.where((hh - ll) > 0, chop.values, 50.0)
        chop = np.where(np.isnan(chop), 50.0, chop)
        return chop
    
    # RSI - momentum filter
    def rsi(close, window=14):
        close_s = pd.Series(close)
        delta = close_s.diff()
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).rolling(window=window, min_periods=window).mean()
        avg_loss = pd.Series(loss).rolling(window=window, min_periods=window).mean()
        rs = np.where(avg_loss > 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        rsi = np.where(np.isnan(rsi), 50, rsi)
        return rsi.values
    
    # 1d indicators
    kama_1d = kama(close, 10, 2, 30)
    kama_rising = kama_1d > np.roll(kama_1d, 1)
    kama_falling = kama_1d < np.roll(kama_1d, 1)
    kama_rising[0] = False
    kama_falling[0] = False
    
    rsi_1d = rsi(close, 14)
    chop_1d = choppiness_index(high, low, close, 14)
    choppy_market = chop_1d > 61.8  # choppy/range regime
    
    # 1w HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    kama_1w = kama(close_1w, 10, 2, 30)
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_1w)
    kama_1w_uptrend = kama_1w_aligned > np.roll(kama_1w_aligned, 1)
    kama_1w_downtrend = kama_1w_aligned < np.roll(kama_1w_aligned, 1)
    # Handle first values
    kama_1w_uptrend[0] = True  # neutral start
    kama_1w_downtrend[0] = True  # neutral start
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # warmup for indicators
        # Skip if any required data is NaN
        if (np.isnan(kama_1d[i]) or np.isnan(rsi_1d[i]) or 
            np.isnan(chop_1d[i]) or np.isnan(kama_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: KAMA turns down OR chop ends (trend resumes)
            if kama_falling[i] or not choppy_market[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: KAMA turns up OR chop ends (trend resumes)
            if kama_rising[i] or not choppy_market[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Need choppy market for mean reversion
            if choppy_market[i]:
                # Long: KAMA up + RSI > 50 + weekly uptrend
                if kama_rising[i] and rsi_1d[i] > 50 and kama_1w_uptrend[i]:
                    position = 1
                    signals[i] = 0.25
                # Short: KAMA down + RSI < 50 + weekly downtrend
                elif kama_falling[i] and rsi_1d[i] < 50 and kama_1w_downtrend[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals