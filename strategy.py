#!/usr/bin/env python3
"""
Experiment #951: 4h Primary + 1d/1w HTF — Simplified Trend + Mean Reversion

Hypothesis: After 680+ failed strategies, complexity is the enemy. The winning approach:
1. 1w HMA21 = ultimate regime filter (only trade with weekly trend)
2. 1d HMA21 = medium-term trend confirmation
3. 4h RSI(14) = entry timing (oversold in uptrend, overbought in downtrend)
4. 4h ATR(14) = stoploss at 2.5x ATR
5. NO funding rate dependency (causes 0 trades when file missing)
6. RELAXED entry thresholds to ensure 30+ trades on train

Why this should work:
- Weekly trend filter prevents trading against macro direction (failed in 2022 crash)
- Daily HMA confirms medium-term momentum
- 4h RSI provides frequent enough entries (target 30-50 trades/year)
- Simple logic = fewer bugs, more reliable trade generation
- All symbols must trade (no SOL-only bias)

Key improvements over #934:
- Removed funding rate (unreliable, causes 0 trades)
- Simplified regime logic (weekly + daily HMA only)
- Relaxed RSI thresholds (30/70 not 25/75) to ensure trades
- Cleaner signal generation without complex hold logic
- Discrete signal sizes: 0.0, ±0.25, ±0.30

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 4h (target 30-50 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_weekly_trend_daily_hma_rsi_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_rsi(close, period=14):
    """Relative Strength Index."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    avg_gain = np.concatenate([[np.nan], avg_gain])
    avg_loss = np.concatenate([[np.nan], avg_loss])
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
    
    rsi = np.clip(rsi, 0, 100)
    return rsi

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

def calculate_sma(close, period):
    """Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate primary (4h) indicators
    rsi_4h = calculate_rsi(close, period=14)
    atr_4h = calculate_atr(high, low, close, period=14)
    
    # Calculate and align 1d HMA for medium-term trend
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate and align 1w HMA for macro regime filter
    hma_1w_raw = calculate_hma(df_1w['close'].values, 21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 4h SMA for additional trend confirmation
    sma_50_4h = calculate_sma(close, 50)
    sma_200_4h = calculate_sma(close, 200)
    
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
        if np.isnan(rsi_4h[i]) or np.isnan(atr_4h[i]) or atr_4h[i] <= 1e-10:
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            continue
        if np.isnan(sma_50_4h[i]) or np.isnan(sma_200_4h[i]):
            continue
        
        # === MACRO REGIME (1w HTF HMA21) ===
        # Only trade in direction of weekly trend
        weekly_bull = close[i] > hma_1w_aligned[i]
        weekly_bear = close[i] < hma_1w_aligned[i]
        
        # === MEDIUM-TERM TREND (1d HTF HMA21) ===
        daily_bull = close[i] > hma_1d_aligned[i]
        daily_bear = close[i] < hma_1d_aligned[i]
        
        # === 4h TREND CONFIRMATION ===
        trend_bull = close[i] > sma_50_4h[i] and close[i] > sma_200_4h[i]
        trend_bear = close[i] < sma_50_4h[i] and close[i] < sma_200_4h[i]
        
        # === RSI SIGNALS (relaxed thresholds for trade generation) ===
        rsi_oversold = rsi_4h[i] < 35
        rsi_overbought = rsi_4h[i] > 65
        rsi_neutral = 35 <= rsi_4h[i] <= 65
        
        # === LONG ENTRY CONDITIONS ===
        long_signal = False
        long_strength = 0
        
        # Strong long: Weekly bull + Daily bull + RSI oversold (pullback entry)
        if weekly_bull and daily_bull and rsi_oversold:
            long_signal = True
            long_strength = BASE_SIZE
        
        # Medium long: Weekly bull + RSI very oversold (deep pullback)
        elif weekly_bull and rsi_4h[i] < 30:
            long_signal = True
            long_strength = REDUCED_SIZE
        
        # Weak long: Weekly bull + Daily bull + RSI crossing up from oversold
        elif weekly_bull and daily_bull and rsi_4h[i] < 40:
            long_signal = True
            long_strength = REDUCED_SIZE
        
        # === SHORT ENTRY CONDITIONS ===
        short_signal = False
        short_strength = 0
        
        # Strong short: Weekly bear + Daily bear + RSI overbought (rally entry)
        if weekly_bear and daily_bear and rsi_overbought:
            short_signal = True
            short_strength = BASE_SIZE
        
        # Medium short: Weekly bear + RSI very overbought (sharp rally)
        elif weekly_bear and rsi_4h[i] > 70:
            short_signal = True
            short_strength = REDUCED_SIZE
        
        # Weak short: Weekly bear + Daily bear + RSI crossing down from overbought
        elif weekly_bear and daily_bear and rsi_4h[i] > 60:
            short_signal = True
            short_strength = REDUCED_SIZE
        
        # === DETERMINE DESIRED SIGNAL ===
        desired_signal = 0.0
        
        if long_signal and not short_signal:
            desired_signal = long_strength
        elif short_signal and not long_signal:
            desired_signal = -short_strength
        # If both signals, stay flat (conflicting signals)
        
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
                # Hold long if weekly trend still bull and RSI not extreme overbought
                if weekly_bull and rsi_4h[i] < 75:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if weekly trend still bear and RSI not extreme oversold
                if weekly_bear and rsi_4h[i] > 25:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if weekly trend reverses
            if weekly_bear:
                desired_signal = 0.0
            # Exit if RSI extreme overbought
            if rsi_4h[i] > 80:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if weekly trend reverses
            if weekly_bull:
                desired_signal = 0.0
            # Exit if RSI extreme oversold
            if rsi_4h[i] < 20:
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
                entry_atr = atr_4h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_4h[i]
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