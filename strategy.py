#!/usr/bin/env python3
"""
Experiment #057: 15m Primary + 4h HTF — HMA Trend + RSI Pullback + BB Mean Reversion

Hypothesis: After 56 failed experiments, 15m strategies ALL failed with Sharpe=0.000 (ZERO trades).
The problem: entry conditions too strict. This strategy uses LOOSE filters to ensure trades generate.

Key design:
- Timeframe: 15m (target 40-100 trades/year)
- HTF: 4h HMA(21) for trend bias (call ONCE before loop!)
- Entry: 15m RSI(7) pullback + Bollinger Band touch within HTF trend
- Session: prefer 00-12 UTC but not mandatory
- Position size: 0.18 (smaller for 15m frequency)
- Stoploss: 2.5x ATR trailing
- LOOSE filters: RSI(7)<40 for long, RSI(7)>60 for short (very loose!)

Why this might work:
- 4h HMA provides direction without being too restrictive
- RSI(7) is faster than RSI(14), catches more pullbacks
- BB(20,2.0) touch is common enough to generate trades
- Session preference reduces noise but doesn't block entries
- Small position size (0.18) controls drawdown on higher frequency

Target: Sharpe>0.167 (beat current best), trades>=30 train, trades>=3 test, ALL symbols Sharpe>0
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_hma_rsi_bb_pullback_4h_v1"
timeframe = "15m"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma1 = pd.Series(close).ewm(span=half, min_periods=half, adjust=False).mean().values
    wma2 = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    diff = 2.0 * wma1 - wma2
    hma = pd.Series(diff).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean().values
    
    return hma

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

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Bollinger Bands"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    
    return upper, sma, lower

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
    
    # Load HTF data ONCE before loop (CRITICAL - Rule 1)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate and align 4h HMA for trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate 15m indicators
    hma_15m = calculate_hma(close, period=21)
    rsi = calculate_rsi(close, period=7)  # Faster RSI for 15m
    atr = calculate_atr(high, low, close, period=14)
    bb_upper, bb_mid, bb_lower = calculate_bollinger(close, period=20, std_mult=2.0)
    
    signals = np.zeros(n)
    SIZE = 0.18  # 18% position size (smaller for 15m frequency)
    
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
        if np.isnan(hma_15m[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (4h HMA) ===
        htf_bull = close[i] > hma_4h_aligned[i]
        htf_bear = close[i] < hma_4h_aligned[i]
        
        # === 15m HMA TREND ===
        hma_bull = close[i] > hma_15m[i]
        hma_bear = close[i] < hma_15m[i]
        
        # === BOLLINGER BAND POSITION ===
        bb_width = bb_upper[i] - bb_lower[i]
        if bb_width > 1e-10:
            bb_position = (close[i] - bb_lower[i]) / bb_width
        else:
            bb_position = 0.5
        near_lower = bb_position < 0.20  # Within 20% of lower band
        near_upper = bb_position > 0.80  # Within 20% of upper band
        
        # === RSI FILTER (VERY LOOSE - ensure trades) ===
        rsi_oversold = rsi[i] < 40.0  # Very loose (not <30)
        rsi_overbought = rsi[i] > 60.0  # Very loose (not >70)
        
        # === SESSION FILTER (PREFERENCE, NOT REQUIREMENT) ===
        hour_utc = (prices['open_time'].iloc[i] // 3600000) % 24
        preferred_session = 0 <= hour_utc < 12
        
        # === DESIRED SIGNAL (LOOSE CONDITIONS) ===
        desired_signal = 0.0
        
        # LONG: HTF bull OR 15m bull + pullback to BB lower + RSI oversold
        if (htf_bull or hma_bull) and near_lower and rsi_oversold:
            desired_signal = SIZE
            if not preferred_session:
                desired_signal = SIZE * 0.8  # Slightly reduced outside session
        
        # SHORT: HTF bear OR 15m bear + rally to BB upper + RSI overbought
        elif (htf_bear or hma_bear) and near_upper and rsi_overbought:
            desired_signal = -SIZE
            if not preferred_session:
                desired_signal = -SIZE * 0.8
        
        # FALLBACK 1: Strong RSI extremes (ensure trades generate)
        elif rsi[i] < 25.0:
            desired_signal = SIZE * 0.7
        elif rsi[i] > 75.0:
            desired_signal = -SIZE * 0.7
        
        # FALLBACK 2: BB extreme with any trend alignment
        elif bb_position < 0.10 and (htf_bull or hma_bull):
            desired_signal = SIZE * 0.6
        elif bb_position > 0.90 and (htf_bear or hma_bear):
            desired_signal = -SIZE * 0.6
        
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
        elif desired_signal >= SIZE * 0.5:
            final_signal = SIZE * 0.7
        elif desired_signal <= -SIZE * 0.5:
            final_signal = -SIZE * 0.7
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