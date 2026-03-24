#!/usr/bin/env python3
"""
Experiment #038: 4h Primary + 1d HTF — Simplified Donchian Breakout + HMA Trend + RSI

Hypothesis: After analyzing 37 failed experiments, the pattern is clear:
- Complex regime switching (Choppiness Index) causes 0 trades in many cases
- Too many filters = no signals generated (experiments #029, #030, #037 all Sharpe=0.000)
- SOLUTION: Simplify logic, use proven Donchian+HMA+RSI pattern that worked on SOL (+0.782)
- 1d HMA provides major trend bias (loose filter, not hard requirement)
- 4h HMA for local trend direction
- Donchian(20) breakout as main entry trigger (common enough for trades)
- RSI(14) loose filter (30-70 range) to avoid extreme entries
- ATR(14) trailing stop at 2.5x for risk management
- SIZE = 0.30 (30% position, discrete levels to minimize fee churn)

Key design choices:
- Timeframe: 4h (20-50 trades/year target, proven to work best)
- HTF: 1d HMA(50) for major trend bias (loose filter only)
- Entry: Donchian(20) breakout + HMA alignment + RSI filter
- NO Choppiness Index (causes too many 0-trade failures)
- LOOSE filters to ensure >=30 trades on train, >=3 on test
- Position size: 0.30 (30% of capital, conservative)
- Stoploss: 2.5x ATR trailing

Target: Sharpe>0.167 (beat current best), DD>-40%, trades>=30 on train, trades>=3 on test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_hma_rsi_1d_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - faster response than EMA"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma1 = pd.Series(close).ewm(span=half, min_periods=half, adjust=False).mean().values
    wma2 = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    diff = 2.0 * wma1 - wma2
    hma = pd.Series(diff).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean().values
    
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
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
    
    rsi = np.zeros(n)
    rsi[:] = np.nan
    for i in range(period, n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range for stoploss"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_donchian(high, low, period=20):
    """Donchian Channel - highest high and lowest low over period"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = np.zeros(n)
    lower = np.zeros(n)
    upper[:] = np.nan
    lower[:] = np.nan
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i-period+1:i+1])
        lower[i] = np.min(low[i-period+1:i+1])
    
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d HMA for major trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (4h) indicators
    hma_4h = calculate_hma(close, period=21)
    hma_4h_fast = calculate_hma(close, period=10)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
    signals = np.zeros(n)
    SIZE = 0.30  # 30% position size
    
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
        if np.isnan(hma_4h[i]) or np.isnan(rsi[i]):
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
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1d HMA) - LOOSE FILTER ===
        htf_bull = close[i] > hma_1d_aligned[i]
        htf_bear = close[i] < hma_1d_aligned[i]
        
        # === 4h HMA TREND ===
        hma_bull = close[i] > hma_4h[i]
        hma_bear = close[i] < hma_4h[i]
        hma_fast_above_slow = hma_4h_fast[i] > hma_4h[i]
        hma_fast_below_slow = hma_4h_fast[i] < hma_4h[i]
        
        # === DONCHIAN BREAKOUT SIGNALS ===
        # Breakout above previous upper or below previous lower
        donchian_breakout_bull = close[i] > donchian_upper[i-1]
        donchian_breakout_bear = close[i] < donchian_lower[i-1]
        
        # === RSI FILTER (LOOSE - ensure trades generate) ===
        # Avoid extreme RSI entries but don't block normal trades
        rsi_ok_long = rsi[i] > 30.0 and rsi[i] < 80.0
        rsi_ok_short = rsi[i] < 70.0 and rsi[i] > 20.0
        rsi_momentum_long = rsi[i] > 45.0
        rsi_momentum_short = rsi[i] < 55.0
        
        # === ENTRY SIGNALS (SIMPLIFIED - prioritize trade generation) ===
        desired_signal = 0.0
        
        # LONG entries (multiple conditions, any can trigger)
        long_score = 0
        
        # Condition 1: Donchian breakout + HMA bull + RSI ok
        if donchian_breakout_bull and hma_bull and rsi_ok_long:
            long_score += 2
        
        # Condition 2: HMA crossover + RSI momentum + HTF not bear
        if hma_fast_above_slow and rsi_momentum_long and not htf_bear:
            long_score += 1
        
        # Condition 3: Price above both HMA + HTF bull (trend continuation)
        if hma_bull and htf_bull and rsi[i] > 40.0:
            long_score += 1
        
        # SHORT entries
        short_score = 0
        
        # Condition 1: Donchian breakdown + HMA bear + RSI ok
        if donchian_breakout_bear and hma_bear and rsi_ok_short:
            short_score += 2
        
        # Condition 2: HMA crossover down + RSI momentum + HTF not bull
        if hma_fast_below_slow and rsi_momentum_short and not htf_bull:
            short_score += 1
        
        # Condition 3: Price below both HMA + HTF bear (trend continuation)
        if hma_bear and htf_bear and rsi[i] < 60.0:
            short_score += 1
        
        # Generate signal based on scores
        if long_score >= 2:
            desired_signal = SIZE
        elif short_score >= 2:
            desired_signal = -SIZE
        elif long_score >= 1 and not htf_bear:
            desired_signal = SIZE * 0.5
        elif short_score >= 1 and not htf_bull:
            desired_signal = -SIZE * 0.5
        
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
        elif desired_signal >= SIZE * 0.5:
            final_signal = SIZE * 0.5
        elif desired_signal <= -SIZE * 0.5:
            final_signal = -SIZE * 0.5
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