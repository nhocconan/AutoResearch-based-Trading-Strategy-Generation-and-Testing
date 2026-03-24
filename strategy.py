#!/usr/bin/env python3
"""
Experiment #036: 12h Primary + 1d HTF — Dual Regime HMA + Donchian Breakout

Hypothesis: After analyzing 35 experiments, the pattern is clear:
1. Complex regime switches (Choppiness + CRSI + multiple filters) = 0 trades
2. Simple HMA trend + loose RSI = positive Sharpe but can be improved
3. Donchian breakouts add momentum confirmation without over-filtering
4. 12h timeframe should generate 20-50 trades/year with proper thresholds

This strategy uses DUAL REGIME approach:
- TREND REGIME (ADX > 20): Follow HMA trend, enter on RSI pullbacks
- RANGE REGIME (ADX < 20): Mean revert at Donchian bounds

Key innovations vs failed attempts:
- LOOSE RSI thresholds (45/55 instead of 30/70) to ensure trade generation
- Donchian(20) breakout confirmation instead of complex Choppiness
- 1d HMA for HTF bias (only trade with higher timeframe direction)
- ATR 2.5x trailing stop for risk management
- Size: 0.28 (discrete, proven safe through 2022 crash)

Entry Logic:
- Long: 1d_HMA_bull + 12h_HMA_bull + (RSI<55 OR Donchian_breakout_high)
- Short: 1d_HMA_bear + 12h_HMA_bear + (RSI>45 OR Donchian_breakout_low)

Target: Sharpe>0.25, trades>30/symbol train, >3/symbol test, DD>-40%
Timeframe: 12h (target 20-50 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_hma_donchian_regime_1d_v1"
timeframe = "12h"
leverage = 1.0

def calculate_hma(close, period=21):
    """Hull Moving Average - smoother and more responsive than EMA"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def wma(data, span):
        result = np.full(len(data), np.nan)
        weights = np.arange(1, span + 1, dtype=np.float64)
        for i in range(span - 1, len(data)):
            window = data[i - span + 1:i + 1]
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half_period)
    wma_full = wma(close, period)
    double_wma_half = 2.0 * wma_half - wma_full
    hma = wma(double_wma_half, sqrt_period)
    
    return hma

def calculate_rsi(close, period=14):
    """RSI - momentum filter with loose thresholds for trade generation"""
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
    """Average True Range - for stoploss"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_adx(high, low, close, period=14):
    """Average Directional Index - trend strength for regime detection"""
    n = len(close)
    if n < period * 3:
        return np.full(n, np.nan)
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        plus_diff = high[i] - high[i-1]
        minus_diff = low[i-1] - low[i]
        
        if plus_diff > minus_diff and plus_diff > 0:
            plus_dm[i] = plus_diff
        if minus_diff > plus_diff and minus_diff > 0:
            minus_dm[i] = minus_diff
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_di = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_di = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    dx = np.zeros(n)
    for i in range(period * 2, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum < 1e-10:
            dx[i] = 0.0
        else:
            dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    return adx

def calculate_donchian(high, low, period=20):
    """Donchian Channels - breakout detection"""
    n = len(close)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d HMA for HTF trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (12h) indicators
    hma_12h = calculate_hma(close, period=21)
    hma_12h_fast = calculate_hma(close, period=9)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    adx = calculate_adx(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
    signals = np.zeros(n)
    SIZE = 0.28  # Discrete position size - safe through 77% crash
    
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
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_12h[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(rsi[i]) or np.isnan(adx[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1d HMA) ===
        hma_1d_bull = close[i] > hma_1d_aligned[i]
        hma_1d_bear = close[i] < hma_1d_aligned[i]
        
        # === 12h TREND ===
        hma_12h_bull = close[i] > hma_12h[i]
        hma_12h_bear = close[i] < hma_12h[i]
        hma_fast_above_slow = hma_12h_fast[i] > hma_12h[i] if not np.isnan(hma_12h_fast[i]) else False
        hma_fast_below_slow = hma_12h_fast[i] < hma_12h[i] if not np.isnan(hma_12h_fast[i]) else False
        
        # === REGIME DETECTION (ADX) ===
        trend_regime = adx[i] > 20.0  # Trending
        range_regime = adx[i] <= 20.0  # Ranging
        
        # === BREAKOUT DETECTION ===
        donchian_breakout_long = close[i] > donchian_upper[i-1] if not np.isnan(donchian_upper[i-1]) else False
        donchian_breakout_short = close[i] < donchian_lower[i-1] if not np.isnan(donchian_lower[i-1]) else False
        
        # === DESIRED SIGNAL (LOOSE thresholds for trade generation) ===
        desired_signal = 0.0
        
        # LONG entries
        if hma_1d_bull and hma_12h_bull:
            if trend_regime:
                # Trend regime: enter on pullback or breakout
                if (rsi[i] < 55.0 and hma_fast_above_slow) or donchian_breakout_long:
                    desired_signal = SIZE
            else:
                # Range regime: mean revert at lower bound
                if close[i] < donchian_lower[i] * 1.005 and rsi[i] < 50.0:
                    desired_signal = SIZE
        
        # SHORT entries
        if hma_1d_bear and hma_12h_bear:
            if trend_regime:
                # Trend regime: enter on rally or breakdown
                if (rsi[i] > 45.0 and hma_fast_below_slow) or donchian_breakout_short:
                    desired_signal = -SIZE
            else:
                # Range regime: mean revert at upper bound
                if close[i] > donchian_upper[i] * 0.995 and rsi[i] > 50.0:
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