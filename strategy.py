#!/usr/bin/env python3
"""
Experiment #1024: 4h Primary + 12h/1d HTF — HMA Trend + RSI Pullback + Donchian Breakout

Hypothesis: After #1014 failed (Sharpe=-0.741), the Fisher+Choppiness combo was too restrictive.
This strategy returns to proven patterns that generated trades:

1. HMA(16/48) crossover for primary trend (proven in best strategy Sharpe=0.612)
2. 12h HMA21 for medium-term bias, 1d HMA21 for macro filter
3. RSI(14) pullback entries: long when RSI 40-50 in uptrend, short when RSI 50-60 in downtrend
4. Donchian(20) breakout confirmation for momentum entries
5. ATR(14) 2.5x trailing stop for risk management

Why this should work better than #1014:
- SIMPLER conditions = more trades (avoid 0-trade failure mode)
- HMA+RSI proven in current best strategy
- Pullback entries catch continuations (work in both bull/bear)
- Donchian breakout adds momentum confirmation without over-filtering
- Discrete signal sizes (0.0, ±0.25, ±0.30) minimize fee churn

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 4h (target 30-60 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_hma_rsi_donchian_12h1d_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(series, period):
    """Hull Moving Average."""
    series = pd.Series(series)
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = series.rolling(window=half, min_periods=half).mean() * 2
    wma_full = series.rolling(window=period, min_periods=period).mean()
    
    wma_diff = wma_half - wma_full
    hma = wma_diff.rolling(window=sqrt_period, min_periods=sqrt_period).mean()
    
    return hma.values

def calculate_rsi(close, period=14):
    """Relative Strength Index."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rs = np.zeros(n)
    rs[period:] = np.where(avg_loss[period:] > 1e-10, avg_gain[period:] / avg_loss[period:], 100.0)
    
    rsi[period:] = 100.0 - (100.0 / (1.0 + rs[period:]))
    rsi = np.clip(rsi, 0, 100)
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range."""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_donchian(high, low, period=20):
    """Donchian Channel - returns upper and lower bounds."""
    n = len(high)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
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
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 12h HMA21 for medium-term trend
    hma_12h_raw = calculate_hma(df_12h['close'].values, 21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate and align 1d HMA21 for long-term trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (4h) indicators
    hma_16 = calculate_hma(close, 16)
    hma_48 = calculate_hma(close, 48)
    rsi_14 = calculate_rsi(close, 14)
    atr_14 = calculate_atr(high, low, close, 14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    REDUCED_SIZE = 0.20
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(hma_16[i]) or np.isnan(hma_48[i]):
            continue
        if np.isnan(rsi_14[i]):
            continue
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            continue
        if np.isnan(hma_12h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        
        # === TREND DIRECTION (HMA 16/48 crossover) ===
        hma_bull = hma_16[i] > hma_48[i]
        hma_bear = hma_16[i] < hma_48[i]
        
        # === HTF TREND BIAS ===
        # Medium-term: 12h HMA21
        medium_bull = close[i] > hma_12h_aligned[i]
        medium_bear = close[i] < hma_12h_aligned[i]
        
        # Long-term: 1d HMA21
        long_bull = close[i] > hma_1d_aligned[i]
        long_bear = close[i] < hma_1d_aligned[i]
        
        # === MOMENTUM (Donchian breakout) ===
        donchian_breakout_long = close[i] > donchian_upper[i-1] if i > 0 else False
        donchian_breakout_short = close[i] < donchian_lower[i-1] if i > 0 else False
        
        # === RSI PULLBACK LEVELS ===
        rsi_pullback_long = 40.0 <= rsi_14[i] <= 55.0
        rsi_pullback_short = 45.0 <= rsi_14[i] <= 60.0
        rsi_extreme_long = rsi_14[i] < 35.0
        rsi_extreme_short = rsi_14[i] > 65.0
        
        desired_signal = 0.0
        
        # === LONG ENTRIES ===
        # Primary: HMA bullish + RSI pullback + medium-term bullish
        if hma_bull and rsi_pullback_long and medium_bull:
            desired_signal = BASE_SIZE
        # Secondary: HMA bullish + RSI extreme (deep pullback)
        elif hma_bull and rsi_extreme_long:
            desired_signal = REDUCED_SIZE
        # Momentum: HMA bullish + Donchian breakout
        elif hma_bull and donchian_breakout_long and medium_bull:
            desired_signal = REDUCED_SIZE
        
        # === SHORT ENTRIES ===
        # Primary: HMA bearish + RSI pullback + medium-term bearish
        if hma_bear and rsi_pullback_short and medium_bear:
            desired_signal = -BASE_SIZE
        # Secondary: HMA bearish + RSI extreme (rally into resistance)
        elif hma_bear and rsi_extreme_short:
            desired_signal = -REDUCED_SIZE
        # Momentum: HMA bearish + Donchian breakdown
        elif hma_bear and donchian_breakout_short and medium_bear:
            desired_signal = -REDUCED_SIZE
        
        # === MACRO FILTER (1d HMA) ===
        # Reduce position size if against long-term trend
        if desired_signal > 0 and long_bear:
            desired_signal = REDUCED_SIZE
        elif desired_signal < 0 and long_bull:
            desired_signal = -REDUCED_SIZE
        
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
        
        # === HOLD LOGIC — Maintain position if trend intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if HMA still bullish or RSI not extreme overbought
                if hma_bull and rsi_14[i] < 70.0:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if HMA still bearish or RSI not extreme oversold
                if hma_bear and rsi_14[i] > 30.0:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if HMA turns bearish or RSI extreme overbought
            if hma_bear and rsi_14[i] > 70.0:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if HMA turns bullish or RSI extreme oversold
            if hma_bull and rsi_14[i] < 30.0:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = BASE_SIZE if desired_signal >= BASE_SIZE else REDUCED_SIZE
        elif desired_signal < 0:
            desired_signal = -BASE_SIZE if desired_signal <= -BASE_SIZE else -REDUCED_SIZE
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
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