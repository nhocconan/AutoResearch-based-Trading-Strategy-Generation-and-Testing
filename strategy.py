#!/usr/bin/env python3
"""
12h_1d_KAMA_RSI_Chop
- Trend direction via KAMA on 12h
- Entry timing via RSI pullback in trend direction on 12h
- Regime filter: Choppiness Index on 1d < 61.8 (trending market)
- Stops via reverse signal
- Target: 15-30 trades/year per symbol
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Primary 12h data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === HTF: 1d data (loaded once) ===
    df_1d = get_htf_data(prices, '1d')
    
    # === Indicators on 12h (primary) ===
    # KAMA trend (12h)
    def kama(close, er_len=10, fast=2, slow=30):
        change = np.abs(np.diff(close, n=er_len))
        vol = np.sum(np.abs(np.diff(close)), axis=0)
        er = np.where(vol != 0, change / vol, 0)
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        kama = np.full_like(close, np.nan, dtype=float)
        kama[er_len] = close[er_len]
        for i in range(er_len+1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    kama_val = kama(close, 10, 2, 30)
    kama_long = close > kama_val
    kama_short = close < kama_val
    
    # RSI(14) on 12h for pullback entries
    def rsi(close, period=14):
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.full_like(close, np.nan, dtype=float)
        avg_loss = np.full_like(close, np.nan, dtype=float)
        avg_gain[period] = np.nanmean(gain[1:period+1])
        avg_loss[period] = np.nanmean(loss[1:period+1])
        for i in range(period+1, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi_val = rsi(close, 14)
    rsi_oversold = rsi_val < 30
    rsi_overbought = rsi_val > 70
    
    # === HTF: 1d Choppiness Index (regime filter) ===
    def choppiness_index(high, low, close, period=14):
        atr = np.full_like(close, np.nan, dtype=float)
        tr1 = np.abs(high[1:] - low[1:])
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])
        atr = np.full_like(close, np.nan, dtype=float)
        atr[period] = np.nanmean(tr[1:period+1])
        for i in range(period+1, len(close)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        max_high = np.full_like(close, np.nan, dtype=float)
        min_low = np.full_like(close, np.nan, dtype=float)
        for i in range(period, len(close)):
            max_high[i] = np.max(high[i-period+1:i+1])
            min_low[i] = np.min(low[i-period+1:i+1])
        
        sum_atr = np.nansum(atr[1:], axis=0) if len(atr) > 1 else 0
        range_val = max_high - min_low
        chop = 100 * np.log10(sum_atr / range_val) / np.log10(period)
        return chop
    
    chop_1d = choppiness_index(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    trending_market = chop_1d_aligned < 61.8  # trending when chop < 61.8
    
    # === Signal generation ===
    signals = np.zeros(n)
    
    for i in range(30, n):  # warmup for indicators
        # Skip if any data is NaN
        if (np.isnan(kama_val[i]) or np.isnan(rsi_val[i]) or 
            np.isnan(chop_1d_aligned[i])):
            continue
        
        # Long: KAMA uptrend + RSI oversold pullback + trending market
        if (kama_long[i] and rsi_oversold[i] and trending_market[i]):
            signals[i] = 0.25
        
        # Short: KAMA downtrend + RSI overbought pullback + trending market
        elif (kama_short[i] and rsi_overbought[i] and trending_market[i]):
            signals[i] = -0.25
        
        # Exit: reverse signal or chop > 61.8 (range market)
        elif (signals[i-1] == 0.25 and (kama_short[i] or not trending_market[i])) or \
             (signals[i-1] == -0.25 and (kama_long[i] or not trending_market[i])):
            signals[i] = 0.0
        
        # Otherwise hold
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "12h_1d_KAMA_RSI_Chop"
timeframe = "12h"
leverage = 1.0