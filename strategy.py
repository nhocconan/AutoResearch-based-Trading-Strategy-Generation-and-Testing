#!/usr/bin/env python3
"""
Experiment #093: 1d Primary + 1w HTF — Simple Trend Following with RSI Pullback

Hypothesis: After 12 consecutive failures (Sharpe < 0), the problem is OVER-FILTERING.
Recent strategies failed because:
- #081-#092: All negative Sharpe, many with 0 trades
- Too many regime filters = no entries
- Mean reversion doesn't work in crypto (too trending)

This strategy SIMPLIFIES to proven elements:
1. 1w HMA(21) for major trend bias (very slow, reduces whipsaws)
2. 1d RSI(14) pullback entries in trend direction (not extreme thresholds)
3. ATR(14) 2.5x trailing stop for risk management
4. Discrete sizing: 0.30 (conservative, minimizes fee churn)

Why this should work on 1d:
- 1d timeframe = naturally 20-50 trades/year (fee-efficient)
- 1w HMA is extremely slow = only catches major trends
- RSI pullback (40/60 thresholds) = ensures entries happen regularly
- Simple logic = fewer bugs, more reliable execution
- Trend-following works better than mean-reversion in crypto

Entry Logic:
- Long: Price > 1w HMA + RSI(14) crosses above 40 (pullback ending)
- Short: Price < 1w HMA + RSI(14) crosses below 60 (rally ending)
- Size: 0.30 (discrete, conservative)

Risk: 2.5x ATR trailing stop, signal→0 when stopped out
Target: Sharpe>0.4, trades>30/symbol train, >3/symbol test, DD>-40%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_trend_rsi_pullback_1w_v1"
timeframe = "1d"
leverage = 1.0

def calculate_hma(close, period=21):
    """Hull Moving Average - smooth trend indicator"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half_period = max(period // 2, 1)
    sqrt_period = max(int(np.sqrt(period)), 1)
    
    def wma(series, span):
        if len(series) < span:
            return np.full(len(series), np.nan)
        result = np.full(len(series), np.nan)
        weights = np.arange(1, span + 1, dtype=float)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1]
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half_period)
    wma_full = wma(close, period)
    raw_hma = 2.0 * wma_half - wma_full
    hma = wma(raw_hma, sqrt_period)
    
    return hma

def calculate_rsi(close, period=14):
    """RSI - momentum for pullback detection"""
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
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1w HMA for major trend bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (1d) indicators
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    
    # Calculate 1d SMA for additional trend confirmation
    sma_50 = pd.Series(close).rolling(window=50, min_periods=50).mean().values
    
    signals = np.zeros(n)
    SIZE = 0.30  # Discrete position size (conservative)
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    # RSI crossover tracking
    prev_rsi = np.nan
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            prev_rsi = rsi[i] if not np.isnan(rsi[i]) else prev_rsi
            continue
        
        if np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            prev_rsi = rsi[i] if not np.isnan(rsi[i]) else prev_rsi
            continue
        
        if np.isnan(rsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            prev_rsi = prev_rsi
            continue
        
        # === HTF TREND BIAS (1w HMA) ===
        hma_1w_bull = close[i] > hma_1w_aligned[i]
        hma_1w_bear = close[i] < hma_1w_aligned[i]
        
        # === 1D TREND CONFIRMATION (SMA50) ===
        sma_50_bull = not np.isnan(sma_50[i]) and close[i] > sma_50[i]
        sma_50_bear = not np.isnan(sma_50[i]) and close[i] < sma_50[i]
        
        # === RSI PULLBACK DETECTION ===
        # Long: RSI was below 40, now crosses above 40 (pullback ending in uptrend)
        # Short: RSI was above 60, now crosses below 60 (rally ending in downtrend)
        rsi_cross_long = False
        rsi_cross_short = False
        
        if not np.isnan(prev_rsi):
            rsi_cross_long = prev_rsi < 40.0 and rsi[i] >= 40.0
            rsi_cross_short = prev_rsi > 60.0 and rsi[i] <= 60.0
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # Long entry: 1w HMA bull + 1d SMA50 bull + RSI pullback cross
        if hma_1w_bull and sma_50_bull and rsi_cross_long:
            desired_signal = SIZE
        
        # Short entry: 1w HMA bear + 1d SMA50 bear + RSI pullback cross
        elif hma_1w_bear and sma_50_bear and rsi_cross_short:
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
        prev_rsi = rsi[i]
    
    return signals