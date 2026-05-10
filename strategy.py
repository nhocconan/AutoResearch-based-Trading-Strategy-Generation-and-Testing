#!/usr/bin/env python3
# 4H_1D_KAMA_RSI_Chop_Filter
# Hypothesis: KAMA(10) trend direction + RSI(14) pullback + Chop(14) regime filter.
# Long when KAMA up, RSI<40, Chop>61.8 (range). Short when KAMA down, RSI>60, Chop>61.8.
# Uses daily trend filter: only long if close>EMA50 daily, short if close<EMA50 daily.
# Designed for low trade frequency (<30/year) and robustness in bull/bear via daily trend.

name = "4H_1D_KAMA_RSI_Chop_Filter"
timeframe = "4h"
leverage = 1.0

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
    
    # 1d data for trend filter and Chop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Daily EMA50 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Daily Chop: HIGHLOW = ATR(14), SUM = sum(ATR), RANGE = max(high)-min(low) over 14 days
    tr_1d = np.maximum(high_1d[1:] - low_1d[1:], np.maximum(np.abs(high_1d[1:] - close_1d[:-1]), np.abs(low_1d[1:] - close_1d[:-1])))
    tr_1d = np.concatenate([[np.nan], tr_1d])  # align length
    atr_14 = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    sum_atr_14 = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    max_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    range_14 = max_high_14 - min_low_14
    chop = 100 * np.log10(sum_atr_14 / range_14) / np.log10(14)
    chop[range_14 == 0] = 100  # avoid div0
    
    # 4h KAMA: ER = |close - close[10]| / sum|diff| over 10, SC = [ER*(0.6645-0.0645)+0.0645]^2
    close_series = pd.Series(close)
    change = np.abs(close - np.roll(close, 10))
    change[0:10] = np.nan
    dir_diff = np.abs(np.diff(close, prepend=np.nan))
    volatility = pd.Series(dir_diff).rolling(window=10, min_periods=10).sum().values
    er = change / volatility
    er[volatility == 0] = 0
    sc = (er * (0.6645 - 0.0645) + 0.0645) ** 2
    kama = np.full_like(close, np.nan)
    kama[0] = close[0]
    for i in range(1, len(close)):
        if np.isnan(kama[i-1]) or np.isnan(sc[i]):
            kama[i] = close[i]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # 4h RSI(14)
    delta = np.diff(close, prepend=np.nan)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi[avg_loss == 0] = 100
    rsi[avg_gain == 0] = 0
    
    # Align 1d indicators to 4h
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop, additional_delay_bars=0)
    
    # Signals
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or
            np.isnan(ema50_aligned[i]) or np.isnan(chop_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        kama_up = kama[i] > kama[i-1]
        kama_down = kama[i] < kama[i-1]
        daily_uptrend = close[i] > ema50_aligned[i]
        daily_downtrend = close[i] < ema50_aligned[i]
        rsi_oversold = rsi[i] < 40
        rsi_overbought = rsi[i] > 60
        chop_high = chop_aligned[i] > 61.8  # ranging market
        
        if position == 0:
            # Long: KAMA up, RSI oversold, chop high, daily uptrend
            if kama_up and rsi_oversold and chop_high and daily_uptrend:
                signals[i] = 0.25
                position = 1
            # Short: KAMA down, RSI overbought, chop high, daily downtrend
            elif kama_down and rsi_overbought and chop_high and daily_downtrend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: KAMA down or daily downtrend
            if not kama_up or not daily_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: KAMA up or daily uptrend
            if not kama_down or not daily_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals