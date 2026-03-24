#!/usr/bin/env python3
"""
Experiment #931: 6h Primary + 1w/1d HTF — Triple HMA Trend + RSI Pullback + CHOP Regime

Hypothesis: 6h timeframe is underexplored and offers optimal balance between 4h (too noisy)
and 12h (too slow). Weekly HMA provides primary trend bias, daily HMA crossover confirms
intermediate trend, 6h RSI identifies pullback entries. Choppiness Index filters out
ranging markets where trend strategies fail. This triple-confirm approach should reduce
whipsaw while maintaining sufficient trade frequency (25-40 trades/year).

Key innovations:
1. 1w HMA(21) for primary trend bias - extremely slow, reduces false signals
2. 1d HMA(16/48) crossover for intermediate confirmation - validates weekly bias
3. 6h RSI(7) for entry timing - faster RSI catches pullbacks earlier
4. Choppiness Index(14) regime filter - only trend when CHOP<50, mean-revert when CHOP>60
5. ATR(14) 2.5x trailing stop for risk management
6. Discrete sizing: 0.0, ±0.25, ±0.30 to minimize fee churn

Entry conditions (balanced for trades + quality):
- LONG = 1w HMA bull + 1d HMA bull + 6h RSI<55 + CHOP<55 (trending pullback)
- SHORT = 1w HMA bear + 1d HMA bear + 6h RSI>45 + CHOP<55 (trending rally)
- Range mode (CHOP>60): RSI<25 long, RSI>75 short (mean reversion)

Target: Sharpe>0.45, trades>=25 train, trades>=5 test, DD>-40%
Timeframe: 6h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_triple_hma_rsi_chop_1w1d_v1"
timeframe = "6h"
leverage = 1.0

def calculate_hma(close, period):
    """
    Hull Moving Average (HMA)
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    Reduces lag while maintaining smoothness
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = max(1, period // 2)
    sqrt_n = max(1, int(np.sqrt(period)))
    
    def wma(series, span):
        result = np.full(len(series), np.nan)
        weights = np.arange(1, span + 1, dtype=np.float64)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1].astype(np.float64)
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff[i] = 2.0 * wma_half[i] - wma_full[i]
    
    hma = wma(diff, sqrt_n)
    return hma

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
    
    rsi = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        if avg_loss[i] > 1e-10:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
        else:
            rsi[i] = 100.0
    
    return rsi

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

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    Measures market choppy vs trending
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    CHOP > 61.8 = ranging, CHOP < 38.2 = trending
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        atr_sum = np.sum(tr[i-period+1:i+1])
        hh = np.max(high[i-period+1:i+1])
        ll = np.min(low[i-period+1:i+1])
        if hh > ll and atr_sum > 0:
            chop[i] = 100.0 * np.log10(atr_sum / (hh - ll)) / np.log10(period)
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF HMAs
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    hma_1d_16_raw = calculate_hma(df_1d['close'].values, period=16)
    hma_1d_48_raw = calculate_hma(df_1d['close'].values, period=48)
    hma_1d_16_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_16_raw)
    hma_1d_48_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_48_raw)
    
    # Calculate 6h indicators
    rsi_7 = calculate_rsi(close, period=7)
    atr_14 = calculate_atr(high, low, close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    
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
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi_7[i]) or np.isnan(chop_14[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1w_aligned[i]) or np.isnan(hma_1d_16_aligned[i]) or np.isnan(hma_1d_48_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1w HMA) ===
        htf_1w_bull = close[i] > hma_1w_aligned[i]
        htf_1w_bear = close[i] < hma_1w_aligned[i]
        
        # === INTERMEDIATE TREND (1d HMA crossover) ===
        htf_1d_bull = hma_1d_16_aligned[i] > hma_1d_48_aligned[i]
        htf_1d_bear = hma_1d_16_aligned[i] < hma_1d_48_aligned[i]
        
        # === REGIME DETECTION (Choppiness) ===
        is_trending = chop_14[i] < 55.0
        is_ranging = chop_14[i] > 60.0
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # TREND MODE (CHOP < 55): Trade with HTF bias on pullbacks
        if is_trending:
            # LONG: Weekly bull + Daily bull + RSI pullback
            if htf_1w_bull and htf_1d_bull and rsi_7[i] < 55.0:
                desired_signal = SIZE_BASE
                if rsi_7[i] < 45.0:  # Deeper pullback = stronger signal
                    desired_signal = SIZE_STRONG
            
            # SHORT: Weekly bear + Daily bear + RSI rally
            elif htf_1w_bear and htf_1d_bear and rsi_7[i] > 45.0:
                desired_signal = -SIZE_BASE
                if rsi_7[i] > 55.0:  # Higher rally = stronger signal
                    desired_signal = -SIZE_STRONG
        
        # RANGE MODE (CHOP > 60): Mean reversion at extremes
        elif is_ranging:
            if rsi_7[i] < 25.0:  # Oversold in range
                desired_signal = SIZE_BASE
            elif rsi_7[i] > 75.0:  # Overbought in range
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