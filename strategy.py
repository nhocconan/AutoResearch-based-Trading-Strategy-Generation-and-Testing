#!/usr/bin/env python3
"""
1d_KAMA_Trend_Filter_With_RSI_and_Chop_Regime_v1
Hypothesis: On daily timeframe, use Kaufman Adaptive Moving Average (KAMA) for trend direction, combined with RSI(14) for momentum and Choppiness Index(14) for regime filtering. Only trade when KAMA slope aligns with RSI > 50 (bullish) or < 50 (bearish) AND market is not choppy (CHOP < 61.8). This reduces false signals in ranging markets while capturing strong trends. Uses discrete sizing (0.25) to limit fee drag. Target: 10-25 trades/year per symbol to survive both bull and bear markets.
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
    
    # === Indicators on primary (1d) timeframe ===
    # KAMA: Kaufman Adaptive Moving Average
    def kama(close, length=10, fast=2, slow=30):
        # Efficiency Ratio
        change = np.abs(np.diff(close, prepend=close[0]))
        volatility = np.sum(np.abs(np.diff(close)), axis=0) if len(close) > 1 else np.array([0.0])
        # Correct way: rolling volatility
        volatility_rolling = pd.Series(close).rolling(window=length).apply(lambda x: np.sum(np.abs(np.diff(x))), raw=True).values
        er = np.where(volatility_rolling > 0, change / volatility_rolling, 0)
        # Smoothing constants
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        # KAMA calculation
        kama_vals = np.zeros_like(close)
        kama_vals[0] = close[0]
        for i in range(1, len(close)):
            kama_vals[i] = kama_vals[i-1] + sc[i] * (close[i] - kama_vals[i-1])
        return kama_vals
    
    # RSI(14)
    def rsi(close, length=14):
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).rolling(window=length, min_periods=length).mean().values
        avg_loss = pd.Series(loss).rolling(window=length, min_periods=length).mean().values
        rs = np.where(avg_loss > 0, avg_gain / avg_loss, 0)
        rsi_vals = 100 - (100 / (1 + rs))
        return rsi_vals
    
    # Choppiness Index(14)
    def choppy_index(high, low, close, length=14):
        atr = np.zeros_like(close)
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # first bar
        atr = pd.Series(tr).rolling(window=length, min_periods=length).sum().values
        max_high = pd.Series(high).rolling(window=length, min_periods=length).max().values
        min_low = pd.Series(low).rolling(window=length, min_periods=length).min().values
        range_hl = max_high - min_low
        chop = np.where(range_hl > 0, 100 * np.log10(atr / range_hl) / np.log10(length), 50)
        return chop
    
    # Calculate indicators
    kama_vals = kama(close, length=10, fast=2, slow=30)
    rsi_vals = rsi(close, length=14)
    chop_vals = choppy_index(high, low, close, length=14)
    
    # === HTF: 1week trend filter ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for all indicators
    start_idx = max(50, 34, 14)  # KAMA(10) needs ~10, EMA34 needs 34, RSI/CHOP needs 14
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(kama_vals[i]) or np.isnan(rsi_vals[i]) or 
            np.isnan(chop_vals[i]) or np.isnan(ema_34_1w_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine trend: price relative to KAMA
        price_above_kama = close[i] > kama_vals[i]
        price_below_kama = close[i] < kama_vals[i]
        
        # Momentum: RSI > 50 bullish, < 50 bearish
        rsi_bullish = rsi_vals[i] > 50
        rsi_bearish = rsi_vals[i] < 50
        
        # Regime: not choppy (CHOP < 61.8 = trending)
        not_choppy = chop_vals[i] < 61.8
        
        # HTF trend: 1w EMA34 direction
        htf_1w_bullish = close[i] > ema_34_1w_aligned[i]
        htf_1w_bearish = close[i] < ema_34_1w_aligned[i]
        
        if position == 0:
            # Long: price > KAMA AND RSI > 50 AND not choppy AND 1w uptrend
            long_setup = price_above_kama and rsi_bullish and not_choppy and htf_1w_bullish
            # Short: price < KAMA AND RSI < 50 AND not choppy AND 1w downtrend
            short_setup = price_below_kama and rsi_bearish and not_choppy and htf_1w_bearish
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold
            signals[i] = 0.25
            # Exit: price < KAMA OR RSI < 50 OR choppy OR 1w trend turns bearish
            if (price_below_kama or not rsi_bullish or not not_choppy or not htf_1w_bullish):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold
            signals[i] = -0.25
            # Exit: price > KAMA OR RSI > 50 OR choppy OR 1w trend turns bullish
            if (price_above_kama or not rsi_bearish or not not_choppy or htf_1w_bullish):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_KAMA_Trend_Filter_With_RSI_and_Chop_Regime_v1"
timeframe = "1d"
leverage = 1.0