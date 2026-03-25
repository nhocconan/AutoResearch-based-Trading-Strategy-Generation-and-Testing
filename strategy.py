#!/usr/bin/env python3
"""
Experiment #1252: 12h Primary + 1d HTF — KAMA Adaptive Trend + ROC Momentum

Hypothesis: After 1031 failed experiments, the winning pattern is clear:
- KAMA (Kaufman Adaptive Moving Average) outperforms HMA/EMA in crypto
- KAMA adapts smoothing based on market efficiency ratio (ER)
- In chop: KAMA flattens → reduces whipsaw entries
- In trend: KAMA follows price → captures moves
- ROC momentum confirmation filters false breakouts
- 12h timeframe = natural 20-50 trades/year (fee-friendly)

Key insight from #1247 (Sharpe=0.447): KAMA + ROC on 6h worked well.
Scaling to 12h with 1d HTF trend filter should improve further.

Entry logic (LOOSE but with momentum filter):
- LONG: 12h_KAMA(10) > 12h_KAMA(21) AND 1d_price > 1d_KAMA(21) AND ROC(10) > 0
- SHORT: 12h_KAMA(10) < 12h_KAMA(21) AND 1d_price < 1d_KAMA(21) AND ROC(10) < 0

Why this should work:
- KAMA adapts to volatility → fewer false signals in chop
- 1d trend filter → only trade with higher timeframe direction
- ROC momentum → confirms move has strength
- ATR 2.5x trailing stop → controls drawdown
- Discrete sizing (0.0, ±0.25, ±0.30) → minimal fee churn

Target: Sharpe>0.5, trades>=30 train, trades>=5 test, DD>-35%
Timeframe: 12h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_kama_trend_roc_momentum_1d_v1"
timeframe = "12h"
leverage = 1.0

def calculate_kama(close, fast_period=2, slow_period=30, smoothing_period=10):
    """
    Kaufman Adaptive Moving Average (KAMA)
    
    Adapts smoothing based on market efficiency ratio (ER).
    ER = |price change| / sum of absolute price changes
    High ER (trending) → KAMA follows price closely
    Low ER (choppy) → KAMA flattens, reduces whipsaws
    
    Reference: Kaufman, "Trading Systems and Methods"
    """
    n = len(close)
    kama = np.full(n, np.nan, dtype=np.float64)
    
    if n < slow_period + smoothing_period:
        return kama
    
    # Efficiency Ratio (ER)
    er = np.zeros(n, dtype=np.float64)
    for i in range(slow_period, n):
        if not np.isnan(close[i]) and not np.isnan(close[i - slow_period]):
            price_change = abs(close[i] - close[i - slow_period])
            volatility = 0.0
            for j in range(i - slow_period + 1, i + 1):
                if not np.isnan(close[j]) and not np.isnan(close[j - 1]):
                    volatility += abs(close[j] - close[j - 1])
            if volatility > 1e-10:
                er[i] = price_change / volatility
    
    # Smoothing constant
    fast_sc = 2.0 / (fast_period + 1.0)
    slow_sc = 2.0 / (slow_period + 1.0)
    
    # Initialize KAMA at SMA of first smoothing_period bars
    valid_close = []
    for i in range(smoothing_period):
        if not np.isnan(close[i]):
            valid_close.append(close[i])
    if len(valid_close) >= smoothing_period:
        kama[smoothing_period - 1] = np.mean(valid_close[-smoothing_period:])
    else:
        return kama
    
    # Calculate KAMA
    for i in range(smoothing_period, n):
        if np.isnan(er[i]) or np.isnan(kama[i - 1]):
            continue
        
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama

def calculate_roc(close, period=10):
    """
    Rate of Change (ROC) - Momentum indicator
    
    ROC = ((close - close_n_periods_ago) / close_n_periods_ago) * 100
    Positive = upward momentum, Negative = downward momentum
    
    Reference: Standard momentum indicator
    """
    n = len(close)
    roc = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        if not np.isnan(close[i]) and not np.isnan(close[i - period]) and close[i - period] != 0:
            roc[i] = ((close[i] - close[i - period]) / close[i - period]) * 100.0
    
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
    kama_1d_raw = calculate_kama(df_1d['close'].values, fast_period=2, slow_period=30, smoothing_period=10)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_raw)
    
    # Calculate 12h indicators
    kama_12h_fast = calculate_kama(close, fast_period=2, slow_period=10, smoothing_period=5)
    kama_12h_slow = calculate_kama(close, fast_period=2, slow_period=30, smoothing_period=10)
    roc_10 = calculate_roc(close, period=10)
    atr_14 = calculate_atr(high, low, close, period=14)
    
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
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(kama_12h_fast[i]) or np.isnan(kama_12h_slow[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(roc_10[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(kama_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === TREND DIRECTION (Daily KAMA) ===
        price_above_1d = close[i] > kama_1d_aligned[i]
        price_below_1d = close[i] < kama_1d_aligned[i]
        
        # === 12h KAMA TREND ===
        kama_bullish = kama_12h_fast[i] > kama_12h_slow[i]
        kama_bearish = kama_12h_fast[i] < kama_12h_slow[i]
        
        # === MOMENTUM CONFIRMATION (ROC) ===
        roc_positive = roc_10[i] > 0.0
        roc_negative = roc_10[i] < 0.0
        
        # === ENTRY LOGIC (LOOSE with momentum filter) ===
        desired_signal = 0.0
        
        # LONG: 12h KAMA bullish + 1d trend up + ROC positive
        if kama_bullish and price_above_1d and roc_positive:
            # Strong signal: ROC > 2% (strong momentum)
            if roc_10[i] > 2.0:
                desired_signal = SIZE_STRONG
            else:
                desired_signal = SIZE_BASE
        
        # SHORT: 12h KAMA bearish + 1d trend down + ROC negative
        elif kama_bearish and price_below_1d and roc_negative:
            # Strong signal: ROC < -2% (strong momentum)
            if roc_10[i] < -2.0:
                desired_signal = -SIZE_STRONG
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
                entry_atr = atr_14[i]
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