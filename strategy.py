#!/usr/bin/env python3
"""
Experiment #160: 6h Primary + 1d/1w HTF — HMA Trend + RSI Pullback + Vol Filter

Hypothesis: 6h timeframe is underexplored and offers sweet spot between 4h (fee drag)
and 12h (too few trades). Previous 6h experiments failed due to ZERO trades from
overly strict conditions.

Key learnings from 159 failed experiments:
- Complex regime-switching fails on BTC/ETH (#135, #138, #146, #158)
- cRSI didn't work on 6h (#143)
- Woodie pivots failed (#140)
- Strategies with Sharpe=0.000 had ZERO trades (#149, #150, #153, #156, #157, #159)
- Previous 6h attempts (#151, #155) had Sharpe < 0.1

New approach for 6h:
- 6h HMA(21) for trend direction (faster than EMA, less lag than SMA)
- 1d HMA(50) for major trend bias (HTF confirmation)
- 1w HMA(50) for weekly regime filter (only trade with weekly trend)
- RSI(14) with LOOSE thresholds (>45 long, <55 short) to ENSURE trades
- ATR ratio filter (ATR7/ATR30 < 1.8) to avoid extreme volatility entries
- 2.5x ATR trailing stop for risk management
- Position size: 0.28 (28% of capital)

Design for trade generation (CRITICAL - avoid 0 trades):
- LOOSE RSI thresholds (45/55 not 30/70)
- Multiple entry paths (primary + fallback)
- Fallback: enter when ALL HTF align strongly (ignore some filters)
- Target 30-60 trades/year on 6h timeframe

Target: Sharpe>0.167, DD>-40%, trades>=30 train, trades>=3 test ALL symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_hma_rsi_vol_1d1w_v2"
timeframe = "6h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - faster response than EMA"""
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

def calculate_atr_ratio(high, low, close, short_period=7, long_period=30):
    """ATR ratio for volatility filter - avoids entering during extreme vol"""
    atr_short = calculate_atr(high, low, close, short_period)
    atr_long = calculate_atr(high, low, close, long_period)
    
    n = len(close)
    ratio = np.zeros(n)
    ratio[:] = np.nan
    
    for i in range(long_period, n):
        if atr_long[i] > 1e-10:
            ratio[i] = atr_short[i] / atr_long[i]
    
    return ratio

def calculate_sma(close, period):
    """Simple Moving Average"""
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
    
    # Calculate and align 1d HMA for major trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate and align 1w HMA for weekly regime filter
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=50)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (6h) indicators
    hma_6h = calculate_hma(close, period=21)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    atr_ratio = calculate_atr_ratio(high, low, close, 7, 30)
    sma_200 = calculate_sma(close, 200)
    
    signals = np.zeros(n)
    SIZE = 0.28  # 28% position size
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(250, n):  # Start after 200 SMA is ready
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_6h[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(atr_ratio[i]):
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
        if np.isnan(sma_200[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1d HMA) ===
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        # === WEEKLY REGIME (1w HMA) ===
        htf_1w_bull = close[i] > hma_1w_aligned[i]
        htf_1w_bear = close[i] < hma_1w_aligned[i]
        
        # === VOLATILITY FILTER ===
        # Only enter when vol is not extreme (ATR ratio < 1.8)
        vol_ok = atr_ratio[i] < 1.8
        
        # === 6h HMA TREND ===
        hma_bull = close[i] > hma_6h[i]
        hma_bear = close[i] < hma_6h[i]
        
        # === SMA200 FILTER ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === RSI CONFIRMATION (LOOSE to ensure trades) ===
        rsi_ok_long = rsi[i] > 45.0  # not oversold
        rsi_ok_short = rsi[i] < 55.0  # not overbought
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # PRIMARY: All conditions aligned (full size)
        # Long: 6h HMA bull + 1d HMA bull + 1w HMA bull + vol ok + RSI ok + above SMA200
        if hma_bull and htf_1d_bull and htf_1w_bull and vol_ok and rsi_ok_long and above_sma200:
            desired_signal = SIZE
        
        # Short: 6h HMA bear + 1d HMA bear + 1w HMA bear + vol ok + RSI ok + below SMA200
        elif hma_bear and htf_1d_bear and htf_1w_bear and vol_ok and rsi_ok_short and below_sma200:
            desired_signal = -SIZE
        
        # FALLBACK 1: Strong HTF alignment (ignore vol filter) - 80% size
        elif hma_bull and htf_1d_bull and htf_1w_bull and rsi[i] > 50.0 and above_sma200:
            desired_signal = SIZE * 0.8
        
        elif hma_bear and htf_1d_bear and htf_1w_bear and rsi[i] < 50.0 and below_sma200:
            desired_signal = -SIZE * 0.8
        
        # FALLBACK 2: 1d + 6h aligned (ignore 1w) - 60% size
        # This ensures trades even when weekly is choppy
        elif hma_bull and htf_1d_bull and vol_ok and rsi[i] > 48.0:
            desired_signal = SIZE * 0.6
        
        elif hma_bear and htf_1d_bear and vol_ok and rsi[i] < 52.0:
            desired_signal = -SIZE * 0.6
        
        # FALLBACK 3: Very strong 6h momentum (ignore HTF) - 40% size
        # Ensures we get SOME trades even in choppy markets
        elif hma_bull and rsi[i] > 55.0 and vol_ok:
            desired_signal = SIZE * 0.4
        
        elif hma_bear and rsi[i] < 45.0 and vol_ok:
            desired_signal = -SIZE * 0.4
        
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
        if desired_signal >= SIZE * 0.9:
            final_signal = SIZE
        elif desired_signal <= -SIZE * 0.9:
            final_signal = -SIZE
        elif desired_signal >= SIZE * 0.7:
            final_signal = SIZE * 0.8
        elif desired_signal <= -SIZE * 0.7:
            final_signal = -SIZE * 0.8
        elif desired_signal >= SIZE * 0.5:
            final_signal = SIZE * 0.6
        elif desired_signal <= -SIZE * 0.5:
            final_signal = -SIZE * 0.6
        elif desired_signal >= SIZE * 0.3:
            final_signal = SIZE * 0.4
        elif desired_signal <= -SIZE * 0.3:
            final_signal = -SIZE * 0.4
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