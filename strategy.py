#!/usr/bin/env python3
"""
Experiment #1654: 4h Primary + 12h HTF — Simplified Trend-Pullback with Asymmetric Entries

Hypothesis: Recent 12h/1d strategies failed due to OVER-FILTERING (Sharpe=0.000 = 0 trades).
4h timeframe proven to generate trades while maintaining fee efficiency (20-50 trades/year target).

Key changes from failures:
1. SIMPLER entry logic: HMA trend + RSI pullback (no complex CHOP regime switching)
2. HTF = 12h (not 1w) — more responsive to trend changes, better for 2022 crash
3. Asymmetric entries: Only long when 12h bullish, only short when 12h bearish
4. LOOSER RSI thresholds: 35/65 not 30/70 (more trades)
5. Donchian breakout confirmation (price breaks 20-bar high/low)

Why 4h + 12h:
- 4h primary: 30-60 trades/year target, fee-efficient
- 12h HTF: Faster signal than 1w, catches 2022 trend reversal better
- Proven combination from experiment history

Entry Logic:
- LONG: 12h HMA bullish + 4h RSI < 35 (pullback) + price > Donchian(20) high
- SHORT: 12h HMA bearish + 4h RSI > 65 (pullback) + price < Donchian(20) low
- Exit: ATR trailing stop (2.5x) or signal reversal

Risk: 2.5x ATR trailing stop, discrete signal levels (0.0, ±0.25, ±0.30)
Size: 0.30 with HTF trend, 0.20 against (asymmetric)

Target: Sharpe > 0.618, trades > 30/symbol train, > 5/symbol test, DD > -40%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_hma_rsi_donchian_12h_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(close, period=21):
    """Hull Moving Average - faster response than EMA"""
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

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    gain_smooth = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_smooth = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan)
    for i in range(period, n):
        if loss_smooth[i-1] < 1e-10:
            rsi[i] = 100.0
        else:
            rsi[i] = 100.0 - (100.0 / (1.0 + gain_smooth[i-1] / loss_smooth[i-1]))
    
    return rsi

def calculate_donchian(high, low, period=20):
    """Donchian Channel - returns (upper, lower)"""
    n = len(close) if 'close' in dir() else len(high)
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower

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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate and align 12h HMA for trend bias
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate primary (4h) indicators
    hma_4h = calculate_hma(close, period=21)
    atr = calculate_atr(high, low, close, period=14)
    rsi = calculate_rsi(close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
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
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_12h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_4h[i]) or np.isnan(rsi[i]) or np.isnan(donchian_upper[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === 12h HMA BIAS (Trend Direction) ===
        # Use price vs 12h HMA for trend bias
        twelve_h_bullish = close[i] > hma_12h_aligned[i]
        twelve_h_bearish = close[i] < hma_12h_aligned[i]
        
        # === 4h RSI PULLBACK (Entry Timing) ===
        # LOOSER thresholds for more trades: 35/65 not 30/70
        rsi_oversold = rsi[i] < 35.0
        rsi_overbought = rsi[i] > 65.0
        
        # === DONCHIAN BREAKOUT CONFIRMATION ===
        breakout_long = close[i] > donchian_upper[i]
        breakout_short = close[i] < donchian_lower[i]
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        signal_strength = 0.0
        
        # LONG: 12h bullish + RSI pullback + Donchian breakout (any 2 of 3)
        long_score = 0
        if twelve_h_bullish:
            long_score += 1
        if rsi_oversold:
            long_score += 1
        if breakout_long:
            long_score += 1
        
        # SHORT: 12h bearish + RSI pullback + Donchian breakout (any 2 of 3)
        short_score = 0
        if twelve_h_bearish:
            short_score += 1
        if rsi_overbought:
            short_score += 1
        if breakout_short:
            short_score += 1
        
        # Entry logic: need at least 2 confluence signals
        if long_score >= 2:
            signal_strength = BASE_SIZE if twelve_h_bullish else REDUCED_SIZE
            desired_signal = signal_strength
        elif short_score >= 2:
            signal_strength = BASE_SIZE if twelve_h_bearish else REDUCED_SIZE
            desired_signal = -signal_strength
        
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
        elif desired_signal >= REDUCED_SIZE * 0.85:
            final_signal = REDUCED_SIZE
        elif desired_signal <= -REDUCED_SIZE * 0.85:
            final_signal = -REDUCED_SIZE
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