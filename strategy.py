#!/usr/bin/env python3
"""
Experiment #1481: 4h Primary + 1d/1w HTF — Simplified HMA Pullback with Weekly Bias

Hypothesis: After analyzing 1104 failed strategies, the pattern is clear:
1. 1d strategies work best (#1477 Sharpe=0.150, current best Sharpe=0.618)
2. 4h strategies fail when over-filtered (too many conditions = 0 trades)
3. KAMA underperformed vs HMA in #1479 (Sharpe=-0.205)
4. Breakout entries fail in bear/range markets (2022 crash, 2025 bear)

Key insight: Use PULLBACK entries in direction of weekly trend, not breakouts.
This strategy uses:
- 1w HMA for macro trend bias (strongest filter, prevents counter-trend trades)
- 1d HMA for intermediate trend confirmation
- 4h HMA pullback entries (RSI < 50 in uptrend, RSI > 50 in downtrend)
- ATR(14)*2.5 trailing stoploss
- Loose entry conditions to ensure ≥30 trades/train, ≥3 trades/test

Why 4h + 1d + 1w should work:
1. 4h = target 20-50 trades/year (minimal fee drag ~1-2.5%)
2. 1w HMA filter prevents trading against secular trend (critical for 2022/2025)
3. Pullback entries (not breakouts) work better in bear/range markets
4. HMA more responsive than KAMA/EMA for trend detection
5. Discrete signal sizes (0.0, ±0.25, ±0.30) minimize fee churn

Timeframe: 4h
HTF: 1d and 1w (call get_htf_data ONCE before loop!)
Position Size: 0.30 (discrete levels)
Target: 20-50 trades/year, Sharpe > 0.618 (beat current best), ALL symbols Sharpe > 0
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_hma_pullback_1d1w_rsi_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(close, period=21):
    """
    Hull Moving Average - more responsive than EMA/KAMA
    HMA = WMA(2*WMA(n/2) - WMA(n))
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    # Calculate WMA for period/2
    half = period // 2
    if half < 1:
        half = 1
    
    wma_half = pd.Series(close).ewm(span=half, min_periods=half, adjust=False).mean().values
    wma_full = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # HMA formula
    hma_raw = 2.0 * wma_half - wma_full
    hma = pd.Series(hma_raw).ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean().values
    
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    gain_smooth = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_smooth = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan)
    mask = loss_smooth > 1e-10
    rsi[mask] = 100.0 - (100.0 / (1.0 + gain_smooth[mask] / loss_smooth[mask]))
    rsi[loss_smooth <= 1e-10] = 100.0
    rsi[:period] = np.nan
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
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
    """Donchian Channel for breakout confirmation"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.nanmax(high[i - period + 1:i + 1])
        lower[i] = np.nanmin(low[i - period + 1:i + 1])
    
    return upper, lower

def calculate_sma(close, period=50):
    """Simple Moving Average for additional trend filter"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1w HMA for macro trend bias (strongest filter)
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate and align 1d HMA for intermediate trend
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (4h) indicators
    hma_4h = calculate_hma(close, period=21)
    hma_4h_fast = calculate_hma(close, period=10)  # Faster HMA for crossover
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    sma_50 = calculate_sma(close, period=50)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):  # Start after indicators are ready
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(rsi[i]) or np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_1w_aligned[i]) or np.isnan(hma_1d_aligned[i]) or np.isnan(hma_4h[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === MACRO TREND (1w HMA) - strongest bias filter ===
        # Only trade in direction of weekly trend
        weekly_bull = close[i] > hma_1w_aligned[i]
        weekly_bear = close[i] < hma_1w_aligned[i]
        
        # === INTERMEDIATE TREND (1d HMA) - confirmation ===
        daily_bull = close[i] > hma_1d_aligned[i]
        daily_bear = close[i] < hma_1d_aligned[i]
        
        # === PRIMARY TREND (4h HMA) ===
        hma_bull = close[i] > hma_4h[i]
        hma_bear = close[i] < hma_4h[i]
        
        # === HMA CROSSOVER (faster signal) ===
        hma_cross_bull = hma_4h_fast[i] > hma_4h[i]
        hma_cross_bear = hma_4h_fast[i] < hma_4h[i]
        
        # === DONCHIAN BREAKOUT ===
        breakout_high = close[i] > donchian_upper[i-1] if not np.isnan(donchian_upper[i-1]) else False
        breakout_low = close[i] < donchian_lower[i-1] if not np.isnan(donchian_lower[i-1]) else False
        
        # === RSI PULLBACK - LOOSE bands for more trades ===
        # In uptrend: enter on pullback (RSI 40-55)
        # In downtrend: enter on bounce (RSI 45-60)
        rsi_pullback_long = 35.0 < rsi[i] < 55.0
        rsi_pullback_short = 45.0 < rsi[i] < 65.0
        rsi_strong_bull = rsi[i] > 50.0
        rsi_strong_bear = rsi[i] < 50.0
        
        # === SMA 50 FILTER ===
        above_sma50 = close[i] > sma_50[i]
        below_sma50 = close[i] < sma_50[i]
        
        # === DESIRED SIGNAL - PULLBACK ENTRIES (not breakouts) ===
        desired_signal = 0.0
        
        # LONG: Weekly bull + Daily bull + 4h pullback entry
        if weekly_bull and daily_bull:
            # Strong long: HMA cross + RSI pullback + above SMA50
            if hma_cross_bull and rsi_pullback_long and above_sma50:
                desired_signal = BASE_SIZE
            # Medium long: Price > HMA + RSI supportive
            elif hma_bull and rsi[i] > 45.0 and rsi[i] < 60.0:
                desired_signal = BASE_SIZE * 0.7
            # Weak long: Weekly trend only (ensure trades in strong trends)
            elif weekly_bull and hma_bull and rsi[i] > 40.0:
                desired_signal = BASE_SIZE * 0.5
        
        # SHORT: Weekly bear + Daily bear + 4h bounce entry
        elif weekly_bear and daily_bear:
            # Strong short: HMA cross + RSI bounce + below SMA50
            if hma_cross_bear and rsi_pullback_short and below_sma50:
                desired_signal = -BASE_SIZE
            # Medium short: Price < HMA + RSI supportive
            elif hma_bear and rsi[i] > 40.0 and rsi[i] < 55.0:
                desired_signal = -BASE_SIZE * 0.7
            # Weak short: Weekly trend only (ensure trades in strong trends)
            elif weekly_bear and hma_bear and rsi[i] < 60.0:
                desired_signal = -BASE_SIZE * 0.5
        
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
        if desired_signal >= BASE_SIZE * 0.85:
            final_signal = BASE_SIZE
        elif desired_signal >= BASE_SIZE * 0.6:
            final_signal = BASE_SIZE * 0.8
        elif desired_signal >= BASE_SIZE * 0.3:
            final_signal = BASE_SIZE * 0.5
        elif desired_signal <= -BASE_SIZE * 0.85:
            final_signal = -BASE_SIZE
        elif desired_signal <= -BASE_SIZE * 0.6:
            final_signal = -BASE_SIZE * 0.8
        elif desired_signal <= -BASE_SIZE * 0.3:
            final_signal = -BASE_SIZE * 0.5
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