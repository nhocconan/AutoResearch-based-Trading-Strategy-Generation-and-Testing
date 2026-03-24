#!/usr/bin/env python3
"""
Experiment #1611: 4h Primary + 1d/1w HTF — Simplified Trend Pullback with Donchian Confirmation

Hypothesis: After analyzing 11 failed 4h experiments, complexity is the enemy. 
The #1604 strategy had Sharpe=0.017 but only +27.5% return — too many false signals.
This strategy SIMPLIFIES: remove complex regime switching, use cleaner trend-following.

Key innovations:
1. 1w HMA(21) = MAJOR trend filter (bull/bear market regime)
2. 1d HMA(21) = INTERMEDIATE trend confirmation (must align with weekly)
3. 4h RSI(14) pullback entries = enter on dips in uptrend, rallies in downtrend
4. Donchian(20) breakout = momentum confirmation for entry timing
5. ATR(14) 3x trailing stop = looser than 2.5x to avoid whipsaw
6. LOOSE entry thresholds: RSI 35/65 (not 30/70) to ensure trade generation
7. Position size 0.30 — discrete levels to minimize fee churn

Why this should beat Sharpe 0.618:
- Simpler logic = fewer conflicting filters = more trades generated
- 1w HMA filter prevents trading against major trend (critical for 2022 crash)
- RSI pullback + Donchian breakout = high-probability entry timing
- 3x ATR stop avoids premature exits in volatile crypto
- 4h timeframe targets 20-50 trades/year — optimal fee efficiency
- LOOSER RSI thresholds (35/65 vs 30/70) = more trade opportunities

Timeframe: 4h (required)
HTF: 1d HMA + 1w HMA for trend bias (use mtf_data helper - call ONCE before loop)
Target: Sharpe > 0.618, trades > 10/symbol train, > 3/symbol test, DD > -50%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_trend_pullback_donchian_1d1w_hma_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(close, period=21):
    """
    Hull Moving Average - reduces lag while maintaining smoothness
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
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

def calculate_rsi(close, period=14):
    """
    Relative Strength Index - momentum oscillator
    RSI = 100 - 100/(1 + RS), RS = avg_gain/avg_loss
    """
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
    Donchian Channel - breakout indicator
    Upper = highest high over period
    Lower = lowest low over period
    """
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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1d HMA for intermediate trend
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate and align 1w HMA for major trend regime
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (4h) indicators
    atr = calculate_atr(high, low, close, period=14)
    rsi = calculate_rsi(close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):
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
        if np.isnan(rsi[i]) or np.isnan(donchian_upper[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === MAJOR TREND REGIME (1w HMA) ===
        weekly_bull = close[i] > hma_1w_aligned[i]
        weekly_bear = close[i] < hma_1w_aligned[i]
        
        # === INTERMEDIATE TREND (1d HMA) ===
        daily_bull = close[i] > hma_1d_aligned[i]
        daily_bear = close[i] < hma_1d_aligned[i]
        
        # === TREND ALIGNMENT CHECK ===
        # Only trade when 1w and 1d agree (strong trend)
        strong_bull = weekly_bull and daily_bull
        strong_bear = weekly_bear and daily_bear
        
        # === RSI PULLBACK SIGNALS ===
        # Long: RSI pulled back to 35-50 zone in bull trend
        rsi_pullback_long = 35.0 <= rsi[i] <= 55.0
        # Short: RSI rallied to 50-65 zone in bear trend
        rsi_pullback_short = 45.0 <= rsi[i] <= 65.0
        
        # === DONCHIAN BREAKOUT CONFIRMATION ===
        # Long: price breaks above Donchian upper (momentum confirmation)
        donchian_breakout_long = close[i] > donchian_upper[i - 1] if i > 0 else False
        # Short: price breaks below Donchian lower (momentum confirmation)
        donchian_breakout_short = close[i] < donchian_lower[i - 1] if i > 0 else False
        
        # === PRIMARY SIGNAL LOGIC ===
        desired_signal = 0.0
        
        # LONG ENTRY: Strong bull trend + RSI pullback + Donchian breakout
        if strong_bull and rsi_pullback_long and donchian_breakout_long:
            desired_signal = BASE_SIZE
        
        # SHORT ENTRY: Strong bear trend + RSI pullback + Donchian breakout
        elif strong_bear and rsi_pullback_short and donchian_breakout_short:
            desired_signal = -BASE_SIZE
        
        # === STOPLOSS CHECK (Trailing ATR 3x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 3.0 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 3.0 * entry_atr
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