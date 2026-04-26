#!/usr/bin/env python3
"""
12h_KAMA_Trend_RSI_ChopFilter
Hypothesis: 12h KAMA trend direction + RSI(14) extremes (oversold/overbought) + choppiness regime filter (CHOP > 61.8 = range) for mean reversion entries. Works in both bull and bear markets by adapting to regime: in trending markets (CHOP < 38.2) follow KAMA trend, in ranging markets (CHOP > 61.8) mean revert at RSI extremes. Uses discrete position sizing (0.25) to minimize fee churn. Target: 12-37 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:  # Need warmup for KAMA, RSI, CHOP
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for HTF regime filter (choppiness)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d choppiness index (CHOP) - 14 period
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index
    
    # ATR(14)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Sum of TR over 14 periods
    sum_tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Max high - min low over 14 periods
    max_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    range_14 = max_high_14 - min_low_14
    
    # Choppiness Index: CHOP = 100 * log10(sum_tr_14 / range_14) / log10(14)
    # Avoid division by zero
    chop_ratio = np.where(range_14 > 0, sum_tr_14 / range_14, 1.0)
    chop = 100 * np.log10(chop_ratio) / np.log10(14)
    
    # Align CHOP to 12h timeframe (1 bar delay for completed 1d bar)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Load 1d data for HTF trend (optional, but we'll use 12h for primary trend)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h KAMA for trend direction
    close_12h = df_12h['close'].values
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close_12h, n=10))  # 10-period net change
    change = np.concatenate([[np.nan]*10, change])  # align
    
    # Volatility: sum of absolute daily changes over 10 periods
    vol = np.sum(np.abs(np.diff(close_12h, n=1)), axis=0)  # 1-period changes
    vol_10 = pd.Series(vol).rolling(window=10, min_periods=10).sum().values
    vol_10 = np.concatenate([[np.nan]*9, vol_10[:-9]])  # align properly
    
    # ER = |change| / volatility
    er = np.where(vol_10 > 0, np.abs(change) / vol_10, 0)
    
    # Smoothing constants: fastest SC=2/(2+1)=0.67, slowest SC=2/(30+1)=0.0645
    sc = (er * (0.67 - 0.0645) + 0.0645) ** 2
    sc = np.where(np.isnan(sc), 0.0645, sc)  # default to slowest
    
    # KAMA calculation
    kama = np.full_like(close_12h, np.nan)
    kama[9] = close_12h[9]  # start after 10 periods
    for i in range(10, len(close_12h)):
        kama[i] = kama[i-1] + sc[i] * (close_12h[i] - kama[i-1])
    
    # Align KAMA to 12h timeframe (already 12h, but need alignment for look-ahead safety)
    kama_aligned = align_htf_to_ltf(prices, df_12h, kama)
    
    # Calculate RSI(14) on 12h close
    delta = np.diff(close_12h)
    delta = np.concatenate([[np.nan], delta])  # align
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi_aligned = align_htf_to_ltf(prices, df_12h, rsi)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Start after warmup (need 50 for KAMA/RSI, 14 for CHOP)
    start_idx = max(50, 14)
    
    for i in range(start_idx, n):
        # Get current values
        close_val = close[i]
        kama_val = kama_aligned[i]
        rsi_val = rsi_aligned[i]
        chop_val = chop_aligned[i]
        
        # Skip if any data not ready
        if (np.isnan(kama_val) or np.isnan(rsi_val) or np.isnan(chop_val)):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Regime filters
        is_ranging = chop_val > 61.8   # CHOP > 61.8 = ranging market (mean revert)
        is_trending = chop_val < 38.2  # CHOP < 38.2 = trending market (follow trend)
        
        # Mean reversion logic (ranging market): RSI extremes
        long_mr = is_ranging and (rsi_val < 30)  # oversold
        short_mr = is_ranging and (rsi_val > 70) # overbought
        
        # Trend following logic (trending market): KAMA direction
        long_trend = is_trending and (close_val > kama_val)
        short_trend = is_trending and (close_val < kama_val)
        
        # Combine logic: in ranging market use mean reversion, in trending use trend follow
        # In neutral regime (38.2 <= CHOP <= 61.8) no new entries, hold or flat
        long_entry = long_mr or long_trend
        short_entry = short_mr or short_trend
        
        # Exit logic: opposite signal or regime change to extreme opposite
        exit_long = (position == 1) and (short_entry or (is_ranging and rsi_val > 50))
        exit_short = (position == -1) and (long_entry or (is_ranging and rsi_val < 50))
        
        if long_entry and position != 1:
            signals[i] = base_size
            position = 1
        elif short_entry and position != -1:
            signals[i] = -base_size
            position = -1
        elif position == 1 and exit_long:
            signals[i] = 0.0
            position = 0
        elif position == -1 and exit_short:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "12h_KAMA_Trend_RSI_ChopFilter"
timeframe = "12h"
leverage = 1.0