#!/usr/bin/env python3
# 1d_kama_rsi_chop_v2
# Hypothesis: 1d strategy using KAMA trend direction + RSI extremes + chop filter.
# In ranging markets (2025+), mean reversion from RSI extremes works when trend is flat (KAMA slope near zero).
# Volume confirmation not needed on 1d - chop filter and RSI extremes provide sufficient edge.
# Discrete sizing (0.0, ±0.25) minimizes fee churn. Target: 30-100 trades over 4 years.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_kama_rsi_chop_v2"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 1w HTF data for regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # KAMA calculation (ER=10, fast=2, slow=30)
    close_s = pd.Series(close)
    change = abs(close_s.diff(10))
    volatility = close_s.diff().abs().rolling(10, min_periods=10).sum()
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/2 - 1/30) + 1/30) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Align KAMA to 1d timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1w, kama)
    
    # RSI calculation (14-period)
    delta = close_s.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Chop index regime filter (14-period) from 1w data
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    atr_1w = pd.Series(high_1w - low_1w).rolling(window=14, min_periods=14).sum().values
    high_14 = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    low_14 = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    chop_denom = np.log10(atr_1w) * np.log10(14)
    chop_denom = np.where(chop_denom == 0, 1e-10, chop_denom)
    chop = 100 * np.log10((high_14 - low_14) / chop_denom) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade when market is ranging (chop > 61.8)
        chop_regime = chop_aligned[i] > 61.8
        
        # Trend filter: KAMA slope near zero (|close - kama| < 1% of price)
        trend_filter = abs(close[i] - kama_aligned[i]) < (0.01 * close[i])
        
        if position == 1:  # Long position
            # Exit: RSI > 50 (mean reversion complete) or chop breaks down
            if rsi[i] > 50 or not chop_regime:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: RSI < 50 (mean reversion complete) or chop breaks down
            if rsi[i] < 50 or not chop_regime:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            if chop_regime and trend_filter:
                # Long entry: RSI < 30 (oversold) in ranging market
                if rsi[i] < 30:
                    position = 1
                    signals[i] = 0.25
                # Short entry: RSI > 70 (overbought) in ranging market
                elif rsi[i] > 70:
                    position = -1
                    signals[i] = -0.25
    
    return signals