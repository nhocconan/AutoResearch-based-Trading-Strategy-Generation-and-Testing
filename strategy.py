#!/usr/bin/env python3
"""
Experiment #1612: 12h Primary + 1d/1w HTF — Simplified Donchian Breakout + HMA Trend

Hypothesis: After 11 failed experiments with overly complex regime detection,
the key is SIMPLICITY + ensuring trade generation. Most 12h strategies failed
with 0 trades because entry conditions were too restrictive.

This strategy uses:
1. Donchian(20) breakout - proven trend entry (break 20-period high/low)
2. 1d HMA(21) - primary trend bias (only long if price > 1d HMA, short if <)
3. 1w HMA(21) - major regime filter (avoid counter-trend trades)
4. RSI(14) - momentum confirmation (RSI > 45 for long, < 55 for short)
5. ATR(14) 2.5x trailing stop - drawdown control
6. LOOSE entry conditions - ensure 20-50 trades/year on 12h

Why this should work better:
- Fewer confluence filters = more trades generated
- HTF trend bias provides edge without over-filtering
- Donchian breakout is proven on higher timeframes
- 12h targets 20-50 trades/year — optimal for fee efficiency
- Discrete position sizing (0.25) minimizes fee churn

Timeframe: 12h (required for this experiment)
HTF: 1d HMA + 1w HMA for trend bias (use mtf_data helper - call ONCE before loop)
Target: Sharpe > 0.618, trades > 10/symbol train, > 3/symbol test, DD > -50%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_hma_1d1w_rsi_atr_v1"
timeframe = "12h"
leverage = 1.0

def calculate_hma(close, period=21):
    """
    Hull Moving Average - reduces lag while maintaining smoothness
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = max(period // 2, 1)
    sqrt_period = max(int(np.sqrt(period)), 1)
    
    def wma(data, w_period):
        if w_period < 1:
            return np.full(len(data), np.nan)
        result = np.full(len(data), np.nan)
        weights = np.arange(1, w_period + 1, dtype=float)
        weights /= weights.sum()
        for i in range(w_period - 1, len(data)):
            window = data[i - w_period + 1:i + 1]
            if np.any(np.isnan(window)):
                continue
            result[i] = np.sum(window * weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan)
    mask = ~np.isnan(wma_half) & ~np.isnan(wma_full)
    diff[mask] = 2.0 * wma_half[mask] - wma_full[mask]
    
    hma = wma(diff, sqrt_period)
    return hma

def calculate_atr(high, low, close, period=14):
    """Average True Range with proper min_periods"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_rsi(close, period=14):
    """RSI with proper min_periods"""
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

def calculate_donchian(high, low, period=20):
    """
    Donchian Channel - breakout system
    Returns: upper_band, lower_band, middle_band
    """
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    middle = (upper + lower) / 2.0
    return upper, lower, middle

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1d HMA for intermediate trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate and align 1w HMA for major regime filter
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (12h) indicators
    atr = calculate_atr(high, low, close, period=14)
    rsi = calculate_rsi(close, period=14)
    donchian_upper, donchian_lower, donchian_middle = calculate_donchian(high, low, period=20)
    
    # Also track previous bar for breakout detection
    donchian_upper_prev = np.roll(donchian_upper, 1)
    donchian_lower_prev = np.roll(donchian_lower, 1)
    donchian_upper_prev[0] = np.nan
    donchian_lower_prev[0] = np.nan
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25
    
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
        if np.isnan(rsi[i]) or np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === TREND BIAS (1d + 1w HMA) ===
        # Only take longs if price above both HTF HMAs (bullish regime)
        # Only take shorts if price below both HTF HMAs (bearish regime)
        daily_bull = close[i] > hma_1d_aligned[i]
        daily_bear = close[i] < hma_1d_aligned[i]
        weekly_bull = close[i] > hma_1w_aligned[i]
        weekly_bear = close[i] < hma_1w_aligned[i]
        
        # Strong bullish: both 1d and 1w HMA below price
        strong_bull = daily_bull and weekly_bull
        # Strong bearish: both 1d and 1w HMA above price
        strong_bear = daily_bear and weekly_bear
        # Mixed regime - reduce position or stay flat
        mixed_regime = (daily_bull and weekly_bear) or (daily_bear and weekly_bull)
        
        # === DONCHIAN BREAKOUT DETECTION ===
        # Long breakout: price breaks above previous Donchian upper
        long_breakout = close[i] > donchian_upper_prev[i] and close[i-1] <= donchian_upper_prev[i-1] if not np.isnan(donchian_upper_prev[i]) else False
        # Short breakout: price breaks below previous Donchian lower
        short_breakout = close[i] < donchian_lower_prev[i] and close[i-1] >= donchian_lower_prev[i-1] if not np.isnan(donchian_lower_prev[i]) else False
        
        # Alternative: price currently above/below Donchian bands
        above_upper = close[i] > donchian_upper[i]
        below_lower = close[i] < donchian_lower[i]
        
        # === RSI MOMENTUM CONFIRMATION ===
        rsi_bullish = rsi[i] > 45.0  # Not oversold
        rsi_bearish = rsi[i] < 55.0  # Not overbought
        rsi_strong_bull = rsi[i] > 50.0
        rsi_strong_bear = rsi[i] < 50.0
        
        # === PRIMARY SIGNAL LOGIC ===
        desired_signal = 0.0
        
        # LONG ENTRY: Strong bullish regime + Donchian breakout + RSI confirmation
        if strong_bull:
            if (long_breakout or above_upper) and rsi_bullish:
                desired_signal = BASE_SIZE
            # Also enter on pullback to Donchian middle in strong bull
            elif close[i] <= donchian_middle[i] * 1.002 and close[i] > donchian_lower[i] and rsi_strong_bull:
                desired_signal = BASE_SIZE
        
        # SHORT ENTRY: Strong bearish regime + Donchian breakout + RSI confirmation
        elif strong_bear:
            if (short_breakout or below_lower) and rsi_bearish:
                desired_signal = -BASE_SIZE
            # Also enter on pullback to Donchian middle in strong bear
            elif close[i] >= donchian_middle[i] * 0.998 and close[i] < donchian_upper[i] and rsi_strong_bear:
                desired_signal = -BASE_SIZE
        
        # MIXED REGIME: Only hold existing positions, don't enter new
        elif mixed_regime:
            if not in_position:
                desired_signal = 0.0
            else:
                # Keep current position but don't add
                desired_signal = BASE_SIZE if position_side > 0 else -BASE_SIZE
        
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
        elif desired_signal <= -BASE_SIZE * 0.85:
            final_signal = -BASE_SIZE
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