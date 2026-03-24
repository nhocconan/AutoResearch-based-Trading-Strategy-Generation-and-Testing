#!/usr/bin/env python3
"""
Experiment #022: 12h Primary + 1d HTF — Donchian Breakout + HMA Trend + RSI Filter

Hypothesis: After 21 experiments, the key insight is that 12h timeframe with 
Donchian breakouts has shown proven success on SOL (Sharpe +0.782 to +0.879).
The failure mode of recent experiments is TOO MANY FILTERS = 0 trades.

Key improvements:
1. 12h primary timeframe - proven to work (20-50 trades/year target)
2. Donchian(20) breakout - simple, proven entry signal
3. 1d HMA(21) for trend bias - only trade breakouts in trend direction
4. LOOSE RSI thresholds (40/60 vs 30/70) - ensures trades generate
5. Minimal filters - just trend + breakout + RSI confirmation
6. 2.5x ATR trailing stop for risk management

Entry Logic:
- Long: 12h price breaks Donchian(20) high + 1d close > 1d HMA + RSI > 40
- Short: 12h price breaks Donchian(20) low + 1d close < 1d HMA + RSI < 60
- Size: 0.30 (discrete, minimizes fee churn)

Risk: 2.5x ATR trailing stop, signal→0 when stopped out
Target: Sharpe>0.3, trades>30/symbol train, >3/symbol test, DD>-40%
Timeframe: 12h (target 20-50 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_hma_rsi_loose_v1"
timeframe = "12h"
leverage = 1.0

def calculate_hma(close, period=21):
    """
    Hull Moving Average - smoother and more responsive than EMA
    HMA = WMA(2*WMA(n/2) - WMA(n))
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    # Calculate WMA for period/2
    half_period = period // 2
    wma_half = np.full(n, np.nan)
    for i in range(half_period - 1, n):
        weights = np.arange(1, half_period + 1)
        wma_half[i] = np.sum(close[i - half_period + 1:i + 1] * weights) / np.sum(weights)
    
    # Calculate WMA for period
    wma_full = np.full(n, np.nan)
    for i in range(period - 1, n):
        weights = np.arange(1, period + 1)
        wma_full[i] = np.sum(close[i - period + 1:i + 1] * weights) / np.sum(weights)
    
    # Calculate raw HMA
    raw_hma = np.full(n, np.nan)
    sqrt_period = int(np.sqrt(period))
    
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            raw_hma[i] = 2.0 * wma_half[i] - wma_full[i]
    
    # Calculate final HMA (WMA of raw HMA with sqrt(period))
    hma = np.full(n, np.nan)
    for i in range(period - 1 + sqrt_period - 1, n):
        if np.isnan(raw_hma[i]):
            continue
        start_idx = i - sqrt_period + 1
        if start_idx < period - 1:
            continue
        weights = np.arange(1, sqrt_period + 1)
        values = raw_hma[start_idx:i + 1]
        if np.any(np.isnan(values)):
            continue
        hma[i] = np.sum(values * weights) / np.sum(weights)
    
    return hma

def calculate_donchian(high, low, period=20):
    """
    Donchian Channels - breakout indicator
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

def calculate_rsi(close, period=14):
    """RSI - momentum filter with loose thresholds"""
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
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE = 0.30  # Discrete position size
    
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
        if np.isnan(hma_1d_aligned[i]) or np.isnan(donchian_upper[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(rsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === DONCHIAN BREAKOUT DETECTION ===
        # Check if price broke out THIS bar (close > upper or close < lower)
        breakout_long = close[i] > donchian_upper[i]
        breakout_short = close[i] < donchian_lower[i]
        
        # === HMA TREND ALIGNMENT (1d HTF) ===
        hma_1d_bull = close[i] > hma_1d_aligned[i]
        hma_1d_bear = close[i] < hma_1d_aligned[i]
        
        # === RSI FILTER (LOOSE thresholds to ensure trades) ===
        rsi_ok_long = rsi[i] > 40.0  # Not extremely oversold
        rsi_ok_short = rsi[i] < 60.0  # Not extremely overbought
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # Long entry: Donchian breakout + 1d HMA bullish + RSI filter
        if breakout_long and hma_1d_bull and rsi_ok_long:
            desired_signal = SIZE
        
        # Short entry: Donchian breakout + 1d HMA bearish + RSI filter
        elif breakout_short and hma_1d_bear and rsi_ok_short:
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