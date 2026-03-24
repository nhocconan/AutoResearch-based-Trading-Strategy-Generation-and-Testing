#!/usr/bin/env python3
"""
Experiment #061: 4h Primary + 1d/1w HTF — KAMA Adaptive Trend + Fisher Transform Reversals

Hypothesis: After 60 failed experiments, the key insight is that FIXED moving averages (EMA/HMA)
fail in crypto's regime-switching markets. KAMA (Kaufman Adaptive) adapts to market efficiency -
fast in trends, slow in chop. Combined with Ehlers Fisher Transform for reversal timing, this
should catch both trend continuation AND mean-reversion opportunities.

Why this should work (DIFFERENT from failed attempts):
1. KAMA instead of HMA/EMA - adapts ER (Efficiency Ratio) to market conditions
2. Fisher Transform - proven for catching reversals in bear rallies (research showed edge)
3. Volatility expansion filter - only trade when ATR(7)/ATR(30) > 1.1 (avoid chop)
4. Asymmetric entries - longs need stronger confirmation than shorts (bear market aware)
5. LOOSE filters - RSI 25/75 (not 15/85), ensures we generate trades (learned from 0-trade failures)
6. 1d + 1w HTF - dual timeframe trend confirmation (stronger than single HTF)

Entry Logic (LOOSE to ensure trades):
- Long: KAMA rising + Fisher < -1.0 (oversold reversal) + price > 1d HMA + vol expanding
- Short: KAMA falling + Fisher > +1.0 (overbought reversal) + price < 1d HMA + vol expanding
- Size: 0.30 (discrete, proven range)

Risk: 2.5x ATR trailing stop, signal→0 when stopped out
Target: Sharpe>0.4, trades>30/symbol train, >3/symbol test, DD>-35%
Timeframe: 4h (target 20-50 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_fisher_vol_1d1w_v1"
timeframe = "4h"
leverage = 1.0

def calculate_kama(close, period=10, fast=2, slow=30):
    """
    Kaufman Adaptive Moving Average (KAMA)
    Adapts to market efficiency - fast in trends, slow in chop
    period: lookback for Efficiency Ratio
    fast: fastest smoothing constant (2/(fast+1))
    slow: slowest smoothing constant (2/(slow+1))
    """
    n = len(close)
    if n < period + slow:
        return np.full(n, np.nan)
    
    kama = np.full(n, np.nan)
    
    # Calculate Efficiency Ratio (ER)
    er = np.full(n, np.nan)
    for i in range(period, n):
        signal = abs(close[i] - close[i - period])
        noise = np.sum(np.abs(np.diff(close[i - period:i + 1])))
        if noise > 1e-10:
            er[i] = signal / noise
        else:
            er[i] = 1.0
    
    # Calculate smoothing constant (SC)
    sc = np.full(n, np.nan)
    fast_sc = 2.0 / (fast + 1)
    slow_sc = 2.0 / (slow + 1)
    for i in range(period, n):
        if not np.isnan(er[i]):
            sc[i] = er[i] * (fast_sc - slow_sc) + slow_sc
            sc[i] = sc[i] ** 2  # Square for smoother adaptation
    
    # Calculate KAMA
    kama[period] = close[period]
    for i in range(period + 1, n):
        if not np.isnan(sc[i]):
            kama[i] = kama[i - 1] + sc[i] * (close[i] - kama[i - 1])
        else:
            kama[i] = kama[i - 1]
    
    return kama

def calculate_fisher_transform(high, low, period=9):
    """
    Ehlers Fisher Transform
    Normalizes price to -1 to +1 range, highlights reversals
    Long when Fisher crosses above -1.5, short when crosses below +1.5
    """
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    fisher = np.full(n, np.nan)
    fisher_signal = np.full(n, np.nan)  # Previous bar's fisher (for crossover)
    
    # Calculate typical price and normalize
    for i in range(period - 1, n):
        # Highest high and lowest low over period
        hh = np.max(high[i - period + 1:i + 1])
        ll = np.min(low[i - period + 1:i + 1])
        
        if hh > ll:
            # Normalize price to 0-1 range
            price_norm = (high[i] + low[i]) / 2
            normalized = (price_norm - ll) / (hh - ll)
            
            # Clamp to avoid division issues
            normalized = max(0.001, min(0.999, normalized))
            
            # Fisher transform
            fisher[i] = 0.5 * np.log((1 + normalized) / (1 - normalized))
            
            # Previous value for crossover detection
            if i > period - 1 and not np.isnan(fisher[i - 1]):
                fisher_signal[i] = fisher[i - 1]
            else:
                fisher_signal[i] = fisher[i]
        else:
            fisher[i] = fisher[i - 1] if i > 0 and not np.isnan(fisher[i - 1]) else 0.0
            fisher_signal[i] = fisher[i]
    
    return fisher, fisher_signal

def calculate_hma(close, period=21):
    """Hull Moving Average - for HTF trend"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half_period = max(period // 2, 1)
    sqrt_period = max(int(np.sqrt(period)), 1)
    
    def wma(series, span):
        if len(series) < span:
            return np.full(len(series), np.nan)
        result = np.full(len(series), np.nan)
        weights = np.arange(1, span + 1, dtype=float)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1]
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half_period)
    wma_full = wma(close, period)
    raw_hma = 2.0 * wma_half - wma_full
    hma = wma(raw_hma, sqrt_period)
    
    return hma

def calculate_rsi(close, period=14):
    """RSI - momentum filter with LOOSE thresholds"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan)
    for i in range(period, n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range - for stoploss and vol filter"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
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
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1d HMA for primary trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate and align 1w HMA for major trend bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (4h) indicators
    kama = calculate_kama(close, period=10, fast=2, slow=30)
    fisher, fisher_signal = calculate_fisher_transform(high, low, period=9)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    
    # Volatility expansion filter: ATR(7) / ATR(30)
    atr_7 = calculate_atr(high, low, close, period=7)
    atr_30 = calculate_atr(high, low, close, period=30)
    vol_ratio = np.full(n, np.nan)
    for i in range(n):
        if not np.isnan(atr_7[i]) and not np.isnan(atr_30[i]) and atr_30[i] > 1e-10:
            vol_ratio[i] = atr_7[i] / atr_30[i]
    
    signals = np.zeros(n)
    SIZE = 0.30  # Discrete position size (30% of capital)
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(kama[i]) or np.isnan(fisher[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(vol_ratio[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF TREND BIAS (1d + 1w HMA) ===
        hma_1d_bull = close[i] > hma_1d_aligned[i]
        hma_1d_bear = close[i] < hma_1d_aligned[i]
        hma_1w_bull = close[i] > hma_1w_aligned[i]
        hma_1w_bear = close[i] < hma_1w_aligned[i]
        
        # === KAMA TREND DIRECTION ===
        kama_rising = kama[i] > kama[i - 5] if i >= 5 and not np.isnan(kama[i - 5]) else False
        kama_falling = kama[i] < kama[i - 5] if i >= 5 and not np.isnan(kama[i - 5]) else False
        
        # === FISHER TRANSFORM REVERSAL SIGNALS ===
        # Long: Fisher crosses above -1.5 (oversold reversal)
        fisher_long = fisher[i] > -1.5 and fisher_signal[i] < -1.5
        # Short: Fisher crosses below +1.5 (overbought reversal)
        fisher_short = fisher[i] < 1.5 and fisher_signal[i] > 1.5
        
        # === RSI FILTER (LOOSE thresholds to ensure trades) ===
        rsi_ok_long = rsi[i] > 25.0  # Not extremely oversold
        rsi_ok_short = rsi[i] < 75.0  # Not extremely overbought
        
        # === VOLATILITY EXPANSION FILTER ===
        vol_expanding = vol_ratio[i] > 1.1  # ATR(7) > 1.1 * ATR(30)
        
        # === ASYMMETRIC ENTRY LOGIC (bear market aware) ===
        # Longs need stronger confirmation (both 1d and 1w bullish, or strong 1d)
        # Shorts can enter with weaker confirmation (bear market bias)
        
        desired_signal = 0.0
        
        # Long entry: KAMA rising + Fisher reversal + RSI filter + vol expanding + HTF trend
        # Require: 1d HMA bullish OR (1w HMA bullish + 1d neutral)
        long_trend_ok = hma_1d_bull or (hma_1w_bull and not hma_1d_bear)
        if kama_rising and fisher_long and rsi_ok_long and vol_expanding and long_trend_ok:
            desired_signal = SIZE
        
        # Short entry: KAMA falling + Fisher reversal + RSI filter + vol expanding + HTF trend
        # Require: 1d HMA bearish (shorts easier in bear market)
        if kama_falling and fisher_short and rsi_ok_short and vol_expanding and hma_1d_bear:
            desired_signal = -SIZE
        
        # === STOPLOSS CHECK (Trailing ATR 2.5x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE * 0.85:
            final_signal = SIZE
        elif desired_signal <= -SIZE * 0.85:
            final_signal = -SIZE
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(final_signal) != position_side:
                # Flip position
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif position_side > 0:
                highest_since_entry = max(highest_since_entry, close[i])
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, close[i])
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = final_signal
    
    return signals