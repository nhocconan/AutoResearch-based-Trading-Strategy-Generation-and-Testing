#!/usr/bin/env python3
"""
1d_KAMA_Trend_RSI_ChopFilter
Hypothesis: On daily timeframe, use Kaufman Adaptive Moving Average (KAMA) for trend direction, RSI(14) for momentum confirmation, and Choppiness Index (CHOP) to avoid ranging markets. Enter long when price > KAMA, RSI > 50, and CHOP < 61.8 (trending). Enter short when price < KAMA, RSI < 50, and CHOP < 61.8. Uses 1-week EMA50 as higher-timeframe trend filter to avoid counter-trend trades. Discrete position sizing (0.25) to minimize fee drag. Designed for low trade frequency (<25/year) to work in both bull and bear markets via adaptive trend filter and regime avoidance.
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
    
    # Get 1-week data for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1-week EMA(50) for HTF trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # KAMA on daily close
    def calculate_kama(close, er_fast=2, er_slow=30):
        change = np.abs(np.diff(close, prepend=close[0]))
        volatility = np.abs(np.diff(close, prepend=close[0])).rolling(window=10, min_periods=1).sum()
        er = np.where(volatility != 0, change / volatility, 0)
        sc = (er * (2/(er_fast+1) - 2/(er_slow+1)) + 2/(er_slow+1)) ** 2
        kama = np.zeros_like(close)
        kama[0] = close[0]
        for i in range(1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    kama = calculate_kama(close)
    
    # RSI(14)
    def calculate_rsi(close, window=14):
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).rolling(window=window, min_periods=window).mean()
        avg_loss = pd.Series(loss).rolling(window=window, min_periods=window).mean()
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi.values
    
    rsi = calculate_rsi(close, 14)
    
    # Choppiness Index (CHOP) on daily
    def calculate_chop(high, low, close, window=14):
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
        atr_sum = pd.Series(tr).rolling(window=window, min_periods=window).sum()
        hh = pd.Series(high).rolling(window=window, min_periods=window).max()
        ll = pd.Series(low).rolling(window=window, min_periods=window).min()
        chop = 100 * np.log10(atr_sum / (hh - ll)) / np.log10(window)
        return chop.values
    
    chop_14 = calculate_chop(high, low, close, 14)
    
    # Align 1w EMA50 to daily
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of KAMA (needs ~10), RSI (14), CHOP (14), 1w EMA (50)
    start_idx = max(10, 14, 14, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(chop_14[i]) or 
            np.isnan(ema_50_1w_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        kama_val = kama[i]
        rsi_val = rsi[i]
        chop_val = chop_14[i]
        close_val = close[i]
        ema_50_1w_val = ema_50_1w_aligned[i]
        
        # Regime filter: only trade in trending markets (CHOP < 61.8)
        is_trending = chop_val < 61.8
        
        # HTF trend filter: only trade in direction of 1w EMA50
        uptrend_htf = close_val > ema_50_1w_val
        downtrend_htf = close_val < ema_50_1w_val
        
        if position == 0:
            # Long: price > KAMA, RSI > 50, trending regime, HTF uptrend
            long_signal = (close_val > kama_val) and (rsi_val > 50) and is_trending and uptrend_htf
            # Short: price < KAMA, RSI < 50, trending regime, HTF downtrend
            short_signal = (close_val < kama_val) and (rsi_val < 50) and is_trending and downtrend_htf
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price < KAMA OR RSI < 40 OR chop too high (range) OR HTF trend turns down
            if (close_val < kama_val) or (rsi_val < 40) or (chop_val > 61.8) or (close_val < ema_50_1w_val):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price > KAMA OR RSI > 60 OR chop too high (range) OR HTF trend turns up
            if (close_val > kama_val) or (rsi_val > 60) or (chop_val > 61.8) or (close_val > ema_50_1w_val):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_KAMA_Trend_RSI_ChopFilter"
timeframe = "1d"
leverage = 1.0