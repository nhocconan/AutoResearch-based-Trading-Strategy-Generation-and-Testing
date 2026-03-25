#!/usr/bin/env python3
"""
Experiment #1202: 4h Primary + 1d/1w HTF — KAMA Adaptive Trend + RSI Pullback

Hypothesis: After analyzing 960+ failures, the key issues are: (1) over-filtered entries = 0 trades, 
(2) static MAs like HMA/EMA whipsaw in choppy markets. KAMA (Kaufman Adaptive Moving Average) 
adapts smoothing based on market efficiency ratio - smooth in noise, fast in trends.

Strategy Design:
1. 4h KAMA(21) - Primary trend (adaptive, less whipsaw than HMA)
2. 1d HMA(21) - Higher timeframe trend confirmation
3. RSI(14) 30-70 range - Pullback entries (wider than 35-65 to generate MORE trades)
4. ROC(10) momentum filter - Confirm trend has actual momentum
5. 2.5x ATR trailing stop - Risk management

Key improvements over failed strategies:
- RSI range 30-70 (not 35-65) = more entry triggers
- KAMA instead of HMA = adapts to volatility, fewer false signals
- ROC momentum filter = only enter when trend has actual strength
- Weekly HMA for strong conviction sizing (not required for entry)

Target: Sharpe>0.5, trades>=40 train, trades>=5 test, DD>-35%
Timeframe: 4h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_adaptive_rsi_pullback_1d1w_v1"
timeframe = "4h"
leverage = 1.0

def calculate_kama(close, period=21, fast_sc=2.0/3.0, slow_sc=2.0/31.0):
    """
    Kaufman Adaptive Moving Average
    Adapts smoothing based on market efficiency (trend vs noise)
    """
    n = len(close)
    kama = np.full(n, np.nan, dtype=np.float64)
    
    if n < period:
        return kama
    
    # Efficiency Ratio: net change / sum of absolute changes
    er = np.zeros(n, dtype=np.float64)
    for i in range(period - 1, n):
        if not np.isnan(close[i]) and not np.isnan(close[i - period + 1]):
            net_change = abs(close[i] - close[i - period + 1])
            sum_changes = 0.0
            for j in range(i - period + 2, i + 1):
                if not np.isnan(close[j]) and not np.isnan(close[j - 1]):
                    sum_changes += abs(close[j] - close[j - 1])
            if sum_changes > 1e-10:
                er[i] = net_change / sum_changes
    
    # Smoothing constant
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama[period - 1] = close[period - 1]
    for i in range(period, n):
        if not np.isnan(kama[i - 1]) and not np.isnan(close[i]):
            kama[i] = kama[i - 1] + sc[i] * (close[i] - kama[i - 1])
    
    return kama

def calculate_hma(close, period):
    """Hull Moving Average - reduces lag while smoothing"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = max(1, period // 2)
    sqrt_n = max(1, int(np.sqrt(period)))
    
    def wma(series, span):
        result = np.full(len(series), np.nan, dtype=np.float64)
        weights = np.arange(1, span + 1, dtype=np.float64)
        weight_sum = np.sum(weights)
        for i in range(span - 1, len(series)):
            if not np.isnan(series[i]):
                window = series[i - span + 1:i + 1].astype(np.float64)
                if not np.any(np.isnan(window)):
                    result[i] = np.sum(window * weights) / weight_sum
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff[i] = 2.0 * wma_half[i] - wma_full[i]
    
    return wma(diff, sqrt_n)

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

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss != 0)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi[:period] = np.nan
    return rsi

def calculate_roc(close, period=10):
    """Rate of Change - momentum indicator"""
    n = len(close)
    roc = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        if close[i - period] != 0 and not np.isnan(close[i]) and not np.isnan(close[i - period]):
            roc[i] = (close[i] - close[i - period]) / close[i - period] * 100.0
    return roc

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF indicators
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 4h indicators
    kama_21 = calculate_kama(close, period=21)
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    roc_10 = calculate_roc(close, period=10)
    
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
        
        if np.isnan(rsi_14[i]) or np.isnan(kama_21[i]) or np.isnan(roc_10[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === TREND DIRECTION (4h KAMA + 1d HMA) ===
        price_above_kama = close[i] > kama_21[i]
        price_below_kama = close[i] < kama_21[i]
        
        price_above_1d = close[i] > hma_1d_aligned[i]
        price_below_1d = close[i] < hma_1d_aligned[i]
        
        # Weekly HMA for strong conviction
        hma_1w_valid = not np.isnan(hma_1w_aligned[i])
        price_above_1w = hma_1w_valid and close[i] > hma_1w_aligned[i]
        price_below_1w = hma_1w_valid and close[i] < hma_1w_aligned[i]
        
        # === MOMENTUM FILTER ===
        momentum_positive = roc_10[i] > 0.5  # >0.5% gain over 10 bars
        momentum_negative = roc_10[i] < -0.5  # <-0.5% loss over 10 bars
        
        # === ENTRY LOGIC (RSI pullback in trend direction) ===
        desired_signal = 0.0
        rsi = rsi_14[i]
        
        # LONG: Price above KAMA + above 1d HMA + positive momentum + RSI pullback
        if price_above_kama and price_above_1d and momentum_positive:
            if 30.0 <= rsi <= 70.0:  # Wider range = more trades
                if price_above_1w:
                    desired_signal = SIZE_STRONG
                else:
                    desired_signal = SIZE_BASE
        
        # SHORT: Price below KAMA + below 1d HMA + negative momentum + RSI pullback
        elif price_below_kama and price_below_1d and momentum_negative:
            if 30.0 <= rsi <= 70.0:
                if price_below_1w:
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