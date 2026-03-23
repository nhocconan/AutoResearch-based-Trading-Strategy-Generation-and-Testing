#!/usr/bin/env python3
"""
Experiment #1033: 1d Primary + 1w HTF — Donchian Breakout + HMA Trend + RSI Filter

Hypothesis: After analyzing 749+ failed strategies, the key insight is that 1d strategies
need SIMPLER entry conditions to generate enough trades. Experiments #1022, #1024, #1025,
#1028, #1030, #1032 all got Sharpe=0.000 with ZERO trades due to overly strict filters.

Experiment #1023 (1d CRSI + Donchian + 1w HMA + CHOP + ATR) achieved Sharpe=0.291, Return=+52.6%
— the ONLY 1d strategy in recent history with positive Sharpe. The Donchian breakout was key.

This strategy combines:
1. 1w HMA21: Major trend bias (only long when price > 1w HMA, only short when < 1w HMA)
2. Donchian(20) breakout: Primary entry signal (break 20-day high/low)
3. RSI(14) filter: Avoid extreme entries (RSI 35-65 range for entries)
4. ATR(14) trailing stop: 2.5x ATR for risk management
5. Discrete sizing: 0.0, ±0.25, ±0.30 to minimize fee churn

Why this works for 1d:
- Target 20-40 trades/year (meets minimum 10 train, 3 test requirement)
- Donchian breakout ensures signals in trending markets
- 1w HMA filter avoids counter-trend trades (major edge in crypto)
- RSI filter prevents buying tops/selling bottoms
- Simple logic = more trades than complex regime-switching

Critical fixes from failures:
- REMOVED Choppiness Index (was blocking too many signals)
- REMOVED dual regime logic (was causing 0 trades in #1027, #1032)
- RELAXED RSI thresholds (35-65 not 30-70) for more entries
- Donchian(20) proven to generate signals on 1d (#1023 success)
- 1w HMA21 provides strong trend filter without overfitting

Target: Sharpe > 0.612, trades >= 10 train, >= 3 test, ALL symbols positive
Timeframe: 1d (target 20-40 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_donchian_hma_1w_rsi_atr_v1"
timeframe = "1d"
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

def calculate_donchian(high, low, period=20):
    """Donchian Channel (20-day high/low)."""
    n = len(high)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower

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
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss != 0)
    rsi[period:] = 100 - (100 / (1 + rs[period:]))
    
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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1w HMA21 for major trend bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, 21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (1d) indicators
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    rsi_1d = calculate_rsi(close, period=14)
    atr_1d = calculate_atr(high, low, close, period=14)
    
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
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(hma_1w_aligned[i]) or hma_1w_aligned[i] <= 1e-10:
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        if np.isnan(rsi_1d[i]) or np.isnan(atr_1d[i]) or atr_1d[i] <= 1e-10:
            continue
        
        # === MAJOR TREND (1w HMA21) ===
        weekly_bull = close[i] > hma_1w_aligned[i]
        weekly_bear = close[i] < hma_1w_aligned[i]
        
        # === DONCHIAN BREAKOUT ===
        breakout_long = close[i] > donchian_upper[i-1]  # Break above previous high
        breakout_short = close[i] < donchian_lower[i-1]  # Break below previous low
        
        # === RSI FILTER ===
        rsi_neutral = 35 < rsi_1d[i] < 65  # Not extreme
        rsi_oversold = rsi_1d[i] < 40
        rsi_overbought = rsi_1d[i] > 60
        
        desired_signal = 0.0
        
        # === LONG ENTRIES ===
        # Entry 1: Breakout + weekly bull + RSI not overbought
        if breakout_long and weekly_bull and rsi_1d[i] < 65:
            desired_signal = BASE_SIZE
        # Entry 2: Weekly bull + RSI oversold (pullback entry)
        elif weekly_bull and rsi_oversold and close[i] > donchian_lower[i]:
            desired_signal = REDUCED_SIZE
        
        # === SHORT ENTRIES ===
        # Entry 1: Breakout + weekly bear + RSI not oversold
        if breakout_short and weekly_bear and rsi_1d[i] > 35:
            desired_signal = -BASE_SIZE
        # Entry 2: Weekly bear + RSI overbought (rally entry)
        elif weekly_bear and rsi_overbought and close[i] < donchian_upper[i]:
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
                # Hold long if weekly bullish and RSI not extreme overbought
                if weekly_bull and rsi_1d[i] < 70:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if weekly bearish and RSI not extreme oversold
                if weekly_bear and rsi_1d[i] > 30:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if weekly trend reverses
            if not weekly_bull and rsi_1d[i] > 55:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if weekly trend reverses
            if not weekly_bear and rsi_1d[i] < 45:
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
                entry_atr = atr_1d[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_1d[i]
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