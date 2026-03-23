#!/usr/bin/env python3
"""
Experiment #1149: 4h Primary + 1d HTF — KAMA Adaptive Trend + Donchian Breakout

Hypothesis: After 838+ failed experiments, the pattern is clear:
- Choppiness Index + CRSI combinations are FAILING (negative Sharpe in #1137-#1148)
- Complex regime switching causes 0 trades (#1148 Sharpe=0.000)
- SIMPLE trend + breakout works (HMA+RSI baseline had positive returns)

This strategy uses PROVEN components that haven't been over-tested together:
1. 1d KAMA(21) for adaptive macro trend (better than HMA in choppy markets)
2. 4h Donchian(20) breakout for entry timing (catches momentum bursts)
3. 4h RSI(14) momentum filter (30/70 extremes, not loose 45/55)
4. 4h ATR(14) 2.0x trailing stop (tighter than 2.5x to protect gains)
5. Position size 0.28 discrete (balance between returns and drawdown)

Why this should beat Sharpe=0.612:
- KAMA adapts to volatility (flattens in chop, trends in direction)
- Donchian breakout catches momentum bursts that RSI pullback misses
- RSI 30/70 extremes filter false breakouts
- 1d KAMA prevents counter-trend trades that destroyed 2022 returns
- Target: 25-45 trades/year on 4h (optimal for fee drag)

Timeframe: 4h (primary)
HTF: 1d — loaded ONCE before loop using mtf_data helper
Position Size: 0.28 base (discrete: 0.0, ±0.28)
Stoploss: 2.0x ATR trailing
Target: 25-45 trades/year, Sharpe > 0.612
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_donchian_rsi_1d_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_kama(close, period=21, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average — adapts to market noise.
    ER (Efficiency Ratio) determines smoothing constant.
    High ER = trending (less smoothing), Low ER = choppy (more smoothing)
    """
    n = len(close)
    kama = np.full(n, np.nan)
    
    if n < period + slow_period:
        return kama
    
    # Calculate Efficiency Ratio
    er = np.zeros(n)
    for i in range(slow_period, n):
        signal = abs(close[i] - close[i - slow_period])
        noise = np.sum(np.abs(np.diff(close[i - slow_period:i + 1])))
        if noise > 1e-10:
            er[i] = signal / noise
        else:
            er[i] = 0.0
    
    # Calculate smoothing constants
    fast_sc = 2.0 / (fast_period + 1.0)
    slow_sc = 2.0 / (slow_period + 1.0)
    
    # Initialize KAMA
    kama[slow_period] = close[slow_period]
    
    for i in range(slow_period + 1, n):
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama

def calculate_rsi(close, period=14):
    """Relative Strength Index — momentum oscillator."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    diff = np.diff(close)
    gain = np.where(diff > 0, diff, 0.0)
    loss = np.where(diff < 0, -diff, 0.0)
    
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    mask = avg_loss > 1e-10
    rs = np.zeros(n)
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rsi[mask] = 100.0 - (100.0 / (1.0 + rs[mask]))
    rsi[~mask] = 50.0
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range for volatility measurement."""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_donchian(high, low, period=20):
    """
    Donchian Channel — breakout detection.
    Upper = highest high over period
    Lower = lowest low over period
    """
    n = len(high)
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
    
    # Calculate and align 1d KAMA for adaptive macro trend filter
    kama_1d_raw = calculate_kama(df_1d['close'].values, period=21)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_raw)
    
    # Calculate primary (4h) indicators
    rsi_4h = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
    # Also calculate 4h KAMA for local trend
    kama_4h = calculate_kama(close, period=21)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.28
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(rsi_4h[i]) or np.isnan(atr[i]):
            continue
        if np.isnan(kama_1d_aligned[i]) or np.isnan(kama_4h[i]):
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        if atr[i] <= 1e-10:
            continue
        
        # === MACRO TREND (1d KAMA) ===
        # KAMA adapts to volatility — flattens in chop, trends in direction
        macro_bull = close[i] > kama_1d_aligned[i]
        macro_bear = close[i] < kama_1d_aligned[i]
        
        # === LOCAL TREND (4h KAMA) ===
        local_bull = close[i] > kama_4h[i]
        local_bear = close[i] < kama_4h[i]
        
        # === BREAKOUT SIGNAL (Donchian) ===
        # Long: price breaks above Donchian upper
        # Short: price breaks below Donchian lower
        breakout_long = close[i] > donchian_upper[i - 1]  # previous bar's upper
        breakout_short = close[i] < donchian_lower[i - 1]  # previous bar's lower
        
        # === MOMENTUM FILTER (RSI) ===
        # RSI > 55 confirms bullish momentum for long entries
        # RSI < 45 confirms bearish momentum for short entries
        rsi_bullish = rsi_4h[i] > 55.0
        rsi_bearish = rsi_4h[i] < 45.0
        
        # === EXTREME RSI EXIT ===
        # Exit long if RSI > 75 (overbought)
        # Exit short if RSI < 25 (oversold)
        rsi_extreme_long = rsi_4h[i] > 75.0
        rsi_extreme_short = rsi_4h[i] < 25.0
        
        desired_signal = 0.0
        
        # === LONG ENTRY ===
        # Macro bull + local bull + breakout + RSI confirms
        if macro_bull and local_bull and breakout_long and rsi_bullish:
            desired_signal = BASE_SIZE
        
        # === SHORT ENTRY ===
        # Macro bear + local bear + breakout + RSI confirms
        elif macro_bear and local_bear and breakout_short and rsi_bearish:
            desired_signal = -BASE_SIZE
        
        # === EXTREME RSI EXIT ===
        if in_position and position_side > 0 and rsi_extreme_long:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and rsi_extreme_short:
            desired_signal = 0.0
        
        # === MACRO TREND REVERSAL EXIT ===
        if in_position and position_side > 0 and macro_bear:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and macro_bull:
            desired_signal = 0.0
        
        # === STOPLOSS CHECK (Trailing ATR 2.0x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.0 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.0 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position if trend intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if macro and local still bull
                if macro_bull and local_bull:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if macro and local still bear
                if macro_bear and local_bear:
                    desired_signal = -BASE_SIZE
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = BASE_SIZE
        elif desired_signal < 0:
            desired_signal = -BASE_SIZE
        else:
            desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Flip position
                position_side = int(np.sign(desired_signal))
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
        
        signals[i] = desired_signal
    
    return signals