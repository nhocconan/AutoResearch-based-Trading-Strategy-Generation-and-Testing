#!/usr/bin/env python3
"""
1d_KAMA_Direction_RSI_ChopFilter_v1
Hypothesis: On daily timeframe, KAMA trend direction combined with RSI extremes and choppiness regime filter captures sustained moves while avoiding whipsaw in ranging markets. Uses discrete sizing (0.25) to limit trades to 7-25/year. Works in both bull/bear by following adaptive trend (KAMA) and avoiding false signals in chop via Choppiness Index.
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
    
    # Load 1w data ONCE before loop for regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # KAMA (adaptive trend) on 1d
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Efficiency Ratio for KAMA
    close_1d = pd.Series(df_1d['close'])
    change = abs(close_1d - close_1d.shift(10)).values
    volatility = abs(close_1d.diff()).rolling(window=10, min_periods=10).sum().values
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # KAMA calculation
    kama = np.zeros_like(close_1d.values)
    kama[0] = close_1d.iloc[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d.iloc[i] - kama[i-1])
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # RSI(14) on 1d
    delta = close_1d.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi.values)
    
    # Choppiness Index on 1w (regime filter)
    high_1w = pd.Series(df_1w['high'])
    low_1w = pd.Series(df_1w['low'])
    close_1w = pd.Series(df_1w['close'])
    atr_1w = np.maximum(high_1w - low_1w, 
                        np.maximum(abs(high_1w - close_1w.shift(1)), 
                                   abs(low_1w - close_1w.shift(1))))
    atr_sum = atr_1w.rolling(window=14, min_periods=14).sum()
    highest_high = high_1w.rolling(window=14, min_periods=14).max()
    lowest_low = low_1w.rolling(window=14, min_periods=14).min
    chop = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop.values, additional_delay_bars=0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Warmup: max of KAMA (10), RSI (14), Chop (14)
    start_idx = max(10, 14, 14)
    
    for i in range(start_idx, n):
        close_val = close[i]
        kama_val = kama_aligned[i]
        rsi_val = rsi_aligned[i]
        chop_val = chop_aligned[i]
        
        # Skip if any data not ready
        if (np.isnan(kama_val) or np.isnan(rsi_val) or np.isnan(chop_val)):
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        # Regime filter: chop > 61.8 = ranging (mean revert), chop < 38.2 = trending (trend follow)
        is_ranging = chop_val > 61.8
        is_trending = chop_val < 38.2
        
        # Trend direction from KAMA
        is_uptrend = close_val > kama_val
        is_downtrend = close_val < kama_val
        
        # RSI extremes
        rsi_oversold = rsi_val < 30
        rsi_overbought = rsi_val > 70
        
        # Entry conditions
        # In trending regime: follow KAMA direction
        # In ranging regime: mean revert from RSI extremes
        long_condition = False
        short_condition = False
        
        if is_trending:
            # Trend following: long in uptrend, short in downtrend
            long_condition = is_uptrend
            short_condition = is_downtrend
        else:  # ranging regime
            # Mean reversion: long at RSI oversold, short at RSI overbought
            long_condition = rsi_oversold
            short_condition = rsi_overbought
        
        # Exit conditions: opposite signal or regime change to extreme chop
        long_exit = (position == 1 and (not is_uptrend or chop_val > 70))
        short_exit = (position == -1 and (not is_downtrend or chop_val > 70))
        
        if long_condition and position != 1:
            signals[i] = base_size
            position = 1
        elif short_condition and position != -1:
            signals[i] = -base_size
            position = -1
        elif long_exit:
            signals[i] = 0.0
            position = 0
        elif short_exit:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
    
    return signals

name = "1d_KAMA_Direction_RSI_ChopFilter_v1"
timeframe = "1d"
leverage = 1.0