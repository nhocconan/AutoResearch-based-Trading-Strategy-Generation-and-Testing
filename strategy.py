#!/usr/bin/env python3
"""
1d_KAMA_Direction_RSI_Chop_Filter
Hypothesis: Daily KAMA trend direction combined with RSI extremes and choppiness regime filter.
KAMA adapts to market noise, reducing whipsaws in ranging markets. RSI identifies overextended
conditions for mean-reversion entries in choppy regimes. Works in both bull (trend following) and
bear (mean reversion in chop) markets. Target: 15-25 trades/year (60-100 over 4 years) to minimize
fee drag. Uses 1d primary timeframe with 1w HTF trend filter for higher timeframe bias.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for KAMA, RSI, and chop calculation (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    
    # 1d KAMA trend direction (loaded ONCE)
    # KAMA parameters: ER period=10, fast=2, slow=30
    close_1d = df_1d['close'].values
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.abs(np.diff(close_1d)).cumsum()
    volatility = np.concatenate([[volatility[0]], np.diff(volatility)])
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/2 - 2/30) + 2/30) ** 2
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    kama_1d = kama
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    
    # 1d RSI(14) for mean reversion signals
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # 1d Choppiness Index(14) for regime detection
    tr1 = np.abs(high[1:] - low[:-1])
    tr2 = np.abs(high[1:] - close_1d[:-1])
    tr3 = np.abs(low[1:] - close_1d[:-1])
    tr_1d = np.concatenate([[tr1[0]], np.maximum.reduce([tr1[1:], tr2, tr3])])
    atr_sum = pd.Series(tr_1d).rolling(window=14, min_periods=14).sum().values
    hh = pd.Series(high_1d := df_1d['high'].values).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low_1d := df_1d['low'].values).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / (hh - ll)) / np.log10(14)
    chop[hh - ll == 0] = 50  # avoid division by zero
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # 1w HTF trend filter (loaded ONCE)
    df_1w = get_htf_data(prices, '1w')
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    signals = np.zeros(n)
    
    # Start index: need enough for KAMA (10), RSI (14), chop (14), HTF EMA (34)
    start_idx = max(10, 14, 14, 34)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama_1d_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(chop_1d_aligned[i]) or np.isnan(ema_34_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        
        # Regime filter: chop > 61.8 = ranging (mean revert), chop < 38.2 = trending (trend follow)
        is_ranging = chop_1d_aligned[i] > 61.8
        is_trending = chop_1d_aligned[i] < 38.2
        
        if is_ranging:
            # In ranging markets: mean reversion at RSI extremes
            long_entry = rsi_1d_aligned[i] < 30 and curr_close > kama_1d_aligned[i]
            short_entry = rsi_1d_aligned[i] > 70 and curr_close < kama_1d_aligned[i]
        elif is_trending:
            # In trending markets: follow KAMA direction with 1w HTF bias
            long_entry = curr_close > kama_1d_aligned[i] and curr_close > ema_34_1w_aligned[i]
            short_entry = curr_close < kama_1d_aligned[i] and curr_close < ema_34_1w_aligned[i]
        else:
            # Transition regime: no entries
            long_entry = False
            short_entry = False
        
        if long_entry:
            signals[i] = 0.25
        elif short_entry:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "1d_KAMA_Direction_RSI_Chop_Filter"
timeframe = "1d"
leverage = 1.0