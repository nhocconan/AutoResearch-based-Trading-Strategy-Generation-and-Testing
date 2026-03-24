#!/usr/bin/env python3
"""
Experiment #343: 6h Primary + 1d/1w HTF — Weekly Trend + 6h Pullback Strategy

Hypothesis: 6h timeframe captures 2-5 day swings optimally. Previous 6h strategies
failed with 0 trades due to overly strict filters. This version uses:

1. Weekly HMA(21) slope for macro trend (5-bar lookback for stability)
2. Daily HMA(50) for intermediate confirmation
3. 6h RSI(14) pullback to 35-50 (long) or 50-65 (short) - LENIENT for trade freq
4. 6h HMA fast/slow crossover as secondary entry (ensures trades happen)
5. ATR stoploss at 2.5x from entry

Key improvements from failed #340 (Sharpe=-0.605):
- RSI thresholds widened (35-50 vs 30-35) for MORE entry opportunities
- Added HMA crossover as PATH 2 entry (doubles potential signals)
- Weekly trend uses HMA slope not price position (more stable, less whipsaw)
- Simplified logic: 2 entry paths instead of complex regime switching

Target: 40-60 trades/year, Sharpe > 0.40, ALL symbols positive Sharpe
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_weekly_trend_rsi_hma_cross_1d1w_v1"
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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF HMA
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (6h) indicators
    hma_6h_slow = calculate_hma(close, period=21)
    hma_6h_fast = calculate_hma(close, period=10)
    atr = calculate_atr(high, low, close, period=14)
    rsi = calculate_rsi(close, period=14)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_6h_slow[i]) or np.isnan(rsi[i]):
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
        
        # === WEEKLY TREND (slope-based over 5 bars) ===
        weekly_bull = False
        weekly_bear = False
        if i > 5 and not np.isnan(hma_1w_aligned[i-5]):
            weekly_bull = hma_1w_aligned[i] > hma_1w_aligned[i-5]
            weekly_bear = hma_1w_aligned[i] < hma_1w_aligned[i-5]
        
        # === DAILY CONFIRMATION ===
        daily_bull = close[i] > hma_1d_aligned[i]
        daily_bear = close[i] < hma_1d_aligned[i]
        
        # === 6h HMA TREND ===
        hma_bull = close[i] > hma_6h_slow[i]
        hma_bear = close[i] < hma_6h_slow[i]
        
        # === RSI PULLBACK (LENIENT thresholds for trade frequency) ===
        rsi_pullback_long = 35.0 <= rsi[i] <= 50.0
        rsi_pullback_short = 50.0 <= rsi[i] <= 65.0
        
        # === HMA CROSSOVER (fast vs slow) ===
        hma_cross_long = False
        hma_cross_short = False
        if i > 0 and not np.isnan(hma_6h_fast[i]) and not np.isnan(hma_6h_fast[i-1]):
            if not np.isnan(hma_6h_slow[i]) and not np.isnan(hma_6h_slow[i-1]):
                # Fast crosses above slow
                if hma_6h_fast[i-1] <= hma_6h_slow[i-1] and hma_6h_fast[i] > hma_6h_slow[i]:
                    hma_cross_long = True
                # Fast crosses below slow
                if hma_6h_fast[i-1] >= hma_6h_slow[i-1] and hma_6h_fast[i] < hma_6h_slow[i]:
                    hma_cross_short = True
        
        # === ENTRY LOGIC (TWO PATHS for trade frequency) ===
        desired_signal = 0.0
        
        # PATH 1: RSI pullback with weekly + daily trend alignment
        if weekly_bull and daily_bull and rsi_pullback_long:
            desired_signal = SIZE_STRONG if hma_bull else SIZE_BASE
        
        elif weekly_bear and daily_bear and rsi_pullback_short:
            desired_signal = -SIZE_STRONG if hma_bear else -SIZE_BASE
        
        # PATH 2: HMA crossover with daily confirmation (more frequent entries)
        elif hma_cross_long and daily_bull:
            desired_signal = SIZE_BASE
        
        elif hma_cross_short and daily_bear:
            desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (2.5x ATR from entry) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE_STRONG * 0.9:
            final_signal = SIZE_STRONG
        elif desired_signal <= -SIZE_STRONG * 0.9:
            final_signal = -SIZE_STRONG
        elif desired_signal >= SIZE_BASE * 0.9:
            final_signal = SIZE_BASE
        elif desired_signal <= -SIZE_BASE * 0.9:
            final_signal = -SIZE_BASE
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                # New position or flip
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                # Set stoploss
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
        
        signals[i] = final_signal
    
    return signals