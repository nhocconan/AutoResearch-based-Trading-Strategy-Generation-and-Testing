#!/usr/bin/env python3
"""
Experiment #1247: 6h Primary + 1d HTF — KAMA Trend + ROC Momentum + Vol Filter

Hypothesis: After 1027 failed experiments, the pattern is clear:
1. HMA works but whipsaws in ranging markets (BTC/ETH range 70% of time)
2. RSI is too slow for 6h momentum entries
3. CHOP regime filters kill trade frequency → 0 trades
4. Weekly pivot strategies all failed on 6h

NEW APPROACH:
1. KAMA (Kaufman Adaptive) instead of HMA — adapts to volatility, reduces whipsaw
2. ROC(10) instead of RSI — faster momentum signal, crosses 0 frequently
3. ATR ratio (7/30) vol filter — only trade when vol expanding (ATR7/ATR30 > 1.0)
4. LOOSE entries: 1d KAMA trend + ROC direction + vol expanding = entry
5. No CHOP, no complex regime — just trend + momentum + vol

Why 6h might work now:
- 6h is unexplored (0 prior experiments before #1240)
- Between 4h (too many trades) and 12h (too few trades)
- Natural 30-60 trades/year with proper filters
- 1d HTF gives clear trend bias without over-filtering

Entry logic (LOOSE):
- LONG: price > 1d_KAMA AND ROC(10) > 0 AND ATR7/ATR30 > 1.0
- SHORT: price < 1d_KAMA AND ROC(10) < 0 AND ATR7/ATR30 > 1.0

Target: Sharpe>0.5, trades>=30 train, trades>=5 test, DD>-35%
Timeframe: 6h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_kama_trend_roc_momentum_1d_v1"
timeframe = "6h"
leverage = 1.0

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average (KAMA)
    Adapts smoothing based on market efficiency (trend vs noise)
    
    Efficiency Ratio (ER) = |Close - Close_n| / Sum(|Close - Close_prev|)
    Fast SC = 2/(fast+1), Slow SC = 2/(slow+1)
    Smoothing Constant = (ER * (Fast - Slow) + Slow)^2
    """
    n = len(close)
    if n < slow_period + er_period:
        return np.full(n, np.nan)
    
    kama = np.full(n, np.nan, dtype=np.float64)
    
    # Calculate Efficiency Ratio
    er = np.full(n, np.nan, dtype=np.float64)
    for i in range(er_period, n):
        if not np.isnan(close[i]) and not np.isnan(close[i - er_period]):
            signal = abs(close[i] - close[i - er_period])
            noise = 0.0
            for j in range(i - er_period + 1, i + 1):
                noise += abs(close[j] - close[j - 1])
            if noise > 1e-10:
                er[i] = signal / noise
            else:
                er[i] = 1.0
    
    # Calculate KAMA
    fast_sc = 2.0 / (fast_period + 1.0)
    slow_sc = 2.0 / (slow_period + 1.0)
    
    # Initialize KAMA with first valid close
    init_idx = er_period
    while init_idx < n and np.isnan(er[init_idx]):
        init_idx += 1
    if init_idx < n:
        kama[init_idx] = close[init_idx]
    
    for i in range(init_idx + 1, n):
        if np.isnan(er[i]) or np.isnan(kama[i-1]) or np.isnan(close[i]):
            continue
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i-1] + sc * (close[i] - kama[i-1])
    
    return kama

def calculate_roc(close, period=10):
    """Rate of Change — momentum indicator"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    roc = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        if close[i - period] > 1e-10 and not np.isnan(close[i]):
            roc[i] = (close[i] - close[i - period]) / close[i - period] * 100.0
    
    return roc

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF indicators
    kama_1d_raw = calculate_kama(df_1d['close'].values, er_period=10, fast_period=2, slow_period=30)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_raw)
    
    # Calculate 6h indicators
    kama_6h = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    roc_10 = calculate_roc(close, period=10)
    atr_7 = calculate_atr(high, low, close, period=7)
    atr_30 = calculate_atr(high, low, close, period=30)
    
    # ATR ratio for volatility expansion filter
    atr_ratio = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        if not np.isnan(atr_7[i]) and not np.isnan(atr_30[i]) and atr_30[i] > 1e-10:
            atr_ratio[i] = atr_7[i] / atr_30[i]
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup period
    min_bars = 100
    
    for i in range(min_bars, n):
        # Skip if indicators not ready
        if np.isnan(atr_7[i]) or np.isnan(atr_30[i]) or atr_7[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(roc_10[i]) or np.isnan(kama_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === TREND DIRECTION (Daily KAMA) ===
        price_above_1d = close[i] > kama_1d_aligned[i]
        price_below_1d = close[i] < kama_1d_aligned[i]
        
        # === VOLATILITY FILTER (ATR ratio > 1.0 = expanding vol) ===
        vol_expanding = atr_ratio[i] > 1.0
        
        # === MOMENTUM (ROC crossing 0) ===
        roc = roc_10[i]
        roc_positive = roc > 0.0
        roc_negative = roc < 0.0
        
        # === ENTRY LOGIC (LOOSE - guarantee trades) ===
        desired_signal = 0.0
        
        # LONG: Price above 1d KAMA + ROC positive + vol expanding
        if price_above_1d and roc_positive and vol_expanding:
            # Check 6h KAMA slope for confirmation
            if i >= 5 and not np.isnan(kama_6h[i]) and not np.isnan(kama_6h[i-5]):
                kama_slope = kama_6h[i] - kama_6h[i-5]
                if kama_slope > 0:
                    desired_signal = SIZE_STRONG
                else:
                    desired_signal = SIZE_BASE
            else:
                desired_signal = SIZE_BASE
        
        # SHORT: Price below 1d KAMA + ROC negative + vol expanding
        elif price_below_1d and roc_negative and vol_expanding:
            # Check 6h KAMA slope for confirmation
            if i >= 5 and not np.isnan(kama_6h[i]) and not np.isnan(kama_6h[i-5]):
                kama_slope = kama_6h[i] - kama_6h[i-5]
                if kama_slope < 0:
                    desired_signal = -SIZE_STRONG
                else:
                    desired_signal = -SIZE_BASE
            else:
                desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (2.5x ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE_STRONG * 0.9:
            final_signal = SIZE_STRONG
        elif desired_signal <= -SIZE_STRONG * 0.9:
            final_signal = -SIZE_STRONG
        elif desired_signal >= SIZE_BASE * 0.9:
            final_signal = SIZE_BASE
        elif desired_signal <= -SIZE_BASE * 0.9:
            final_signal = -SIZE_BASE
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr_7[i] if not np.isnan(atr_7[i]) else atr_30[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = final_signal
    
    return signals