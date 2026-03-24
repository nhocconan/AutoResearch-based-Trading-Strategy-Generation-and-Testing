#!/usr/bin/env python3
"""
Experiment #107: 1d Primary + 1w HTF — Donchian Breakout + HMA Trend + RSI Filter

Hypothesis: After analyzing 91 failed experiments, the pattern is clear:
- Complex regime filters (Choppiness, dual-regime) cause 0 trades or negative Sharpe
- 1d timeframe with 1w HTF bias shown success (SOL +0.879 with HMA crossover)
- Donchian breakout captures trend continuation better than KAMA crossover alone
- LOOSE RSI thresholds (30/70) ensure trade generation on all symbols
- 1d primary = 20-50 trades/year target (matches proven pattern)

This strategy uses MINIMAL but effective filters:
1. 1w HMA = major trend bias (price above/below)
2. 1d Donchian(20) breakout = entry trigger (proven on SOL +0.782)
3. 1d HMA(21) confirmation = trend alignment
4. RSI loose filter (>30 for long, <70 for short) - ensures trades generate
5. ATR trailing stoploss (2.5x) for risk management
6. NO Choppiness, NO complex regime detection

Key design choices:
- Timeframe: 1d (proven to work, 20-50 trades/year target per instructions)
- HTF: 1w for trend bias (higher than 1d, more stable signal)
- Donchian(20): captures 20-day breakouts, proven edge on crypto
- HMA(21): faster response than EMA, less lag for trend confirmation
- RSI thresholds: 30/70 (loose, ensures trades on BTC/ETH/SOL)
- Position size: 0.28 (28% of capital, conservative for 1d)
- Stoploss: 2.5x ATR trailing (tighter than 3x for better risk control)

Target: Sharpe>0.351, DD>-40%, trades>=30 on train, trades>=3 on test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_donchian_hma_rsi_1w_v1"
timeframe = "1d"
leverage = 1.0

def calculate_hma(close, period=21):
    """
    Hull Moving Average - reduces lag while maintaining smoothness
    HMA = WMA(2*WMA(n/2) - WMA(n))
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    # Calculate WMA for period/2
    half = period // 2
    if half < 1:
        half = 1
    
    wma_half = np.zeros(n)
    wma_half[:] = np.nan
    for i in range(half - 1, n):
        weights = np.arange(1, half + 1)
        wma_half[i] = np.sum(close[i-half+1:i+1] * weights) / np.sum(weights)
    
    # Calculate WMA for full period
    wma_full = np.zeros(n)
    wma_full[:] = np.nan
    for i in range(period - 1, n):
        weights = np.arange(1, period + 1)
        wma_full[i] = np.sum(close[i-period+1:i+1] * weights) / np.sum(weights)
    
    # Calculate HMA
    hma = np.zeros(n)
    hma[:] = np.nan
    sqrt_period = int(np.sqrt(period))
    
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            raw_hma = 2.0 * wma_half[i] - wma_full[i]
            # WMA of raw_hma with sqrt(period)
            if i >= sqrt_period - 1:
                weights = np.arange(1, sqrt_period + 1)
                start_idx = i - sqrt_period + 1
                hma[i] = np.sum(raw_hma[start_idx:i+1] * weights) / np.sum(weights)
    
    return hma

def calculate_donchian(high, low, period=20):
    """
    Donchian Channel - highest high and lowest low over period
    Returns: upper, lower, middle
    """
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    upper = np.zeros(n)
    lower = np.zeros(n)
    upper[:] = np.nan
    lower[:] = np.nan
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i-period+1:i+1])
        lower[i] = np.min(low[i-period+1:i+1])
    
    middle = (upper + lower) / 2.0
    return upper, lower, middle

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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1w HMA for major trend bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (1d) indicators
    hma_1d = calculate_hma(close, period=21)
    donchian_upper, donchian_lower, donchian_mid = calculate_donchian(high, low, period=20)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE = 0.28  # 28% position size (conservative for 1d)
    
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
        if np.isnan(hma_1d[i]) or np.isnan(donchian_upper[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(rsi[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1w HMA) ===
        # Simple: is price above or below weekly HMA?
        htf_bull = close[i] > hma_1w_aligned[i]
        htf_bear = close[i] < hma_1w_aligned[i]
        
        # === 1d TREND (HMA alignment) ===
        hma_bull = close[i] > hma_1d[i]
        hma_bear = close[i] < hma_1d[i]
        
        # === DONCHIAN BREAKOUT ===
        # Long: price breaks above Donchian upper (20-day high)
        # Short: price breaks below Donchian lower (20-day low)
        donchian_breakout_long = close[i] > donchian_upper[i-1]  # breakout above prev high
        donchian_breakout_short = close[i] < donchian_lower[i-1]  # breakout below prev low
        
        # === RSI FILTER (LOOSE - ensure trades generate on all symbols) ===
        # For longs: RSI > 30 (not extremely oversold)
        # For shorts: RSI < 70 (not extremely overbought)
        rsi_ok_long = rsi[i] > 30.0
        rsi_ok_short = rsi[i] < 70.0
        
        # === DESIRED SIGNAL ===
        # LONG: 1w bull + 1d HMA bull + Donchian breakout + RSI > 30
        # SHORT: 1w bear + 1d HMA bear + Donchian breakout + RSI < 70
        desired_signal = 0.0
        
        if htf_bull and hma_bull and donchian_breakout_long and rsi_ok_long:
            desired_signal = SIZE
        elif htf_bear and hma_bear and donchian_breakout_short and rsi_ok_short:
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