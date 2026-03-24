#!/usr/bin/env python3
"""
Experiment #063: 6h Primary + 1d/1w HTF — Weekly Trend + Daily RSI Pullback

Hypothesis: After 62 failed experiments, 6h timeframe needs SIMPLER logic:
- Weekly HMA50 establishes major trend bias (bull/bear regime)
- Daily RSI identifies pullback entries within the trend
- 6h price action for execution timing
- ASYMMETRIC entries: only long in bull regime, only short in bear regime
- This avoids complex regime switching that failed in #060, #062

Why this might work on 6h:
- 6h is middle ground: enough bars for entries, not too many for fee drag
- Weekly trend filter prevents counter-trend trades that destroyed 2022 performance
- Daily RSI pullback is proven pattern (current best uses CRSI on 1d)
- Simple logic = more trades generated (addressing #1 failure mode)

Key design choices:
- Timeframe: 6h (30-60 trades/year target)
- HTF: 1w HMA50 for trend, 1d RSI for entry timing
- Entry: Weekly trend + Daily RSI pullback + 6h momentum confirm
- Position size: 0.27 (27% of capital, conservative for 6h)
- Stoploss: 2.5x ATR trailing
- LOOSE filters to ensure >=30 trades on train, >=3 on test

Target: Sharpe>0.167 (beat current best), DD>-40%, trades>=30 on train, trades>=3 on test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_weekly_trend_daily_rsi_pullback_1d1w_v1"
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
    
    # Calculate and align 1w HMA for major trend bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=50)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate and align 1d RSI for entry timing
    rsi_1d_raw = calculate_rsi(df_1d['close'].values, period=14)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d_raw)
    
    # Calculate primary (6h) indicators
    hma_6h = calculate_hma(close, period=21)
    atr = calculate_atr(high, low, close, period=14)
    sma_50 = calculate_sma(close, period=50)
    
    signals = np.zeros(n)
    SIZE = 0.27  # 27% position size (conservative for 6h)
    
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
        if np.isnan(hma_6h[i]) or np.isnan(sma_50[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_1w_aligned[i]) or np.isnan(rsi_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1w HMA50) ===
        # Weekly trend determines allowed direction
        weekly_bull = close[i] > hma_1w_aligned[i]
        weekly_bear = close[i] < hma_1w_aligned[i]
        
        # === DAILY RSI PULLBACK ===
        # In bull regime: wait for RSI pullback to 35-50 zone
        # In bear regime: wait for RSI rally to 50-65 zone
        daily_rsi = rsi_1d_aligned[i]
        rsi_pullback_long = 30.0 < daily_rsi < 55.0
        rsi_pullback_short = 45.0 < daily_rsi < 70.0
        
        # === 6h MOMENTUM CONFIRM ===
        # Price above 6h HMA for long, below for short
        hma_6h_bull = close[i] > hma_6h[i]
        hma_6h_bear = close[i] < hma_6h[i]
        
        # === SMA50 FILTER (additional trend confirm) ===
        sma_bull = close[i] > sma_50[i]
        sma_bear = close[i] < sma_50[i]
        
        # === DESIRED SIGNAL (Asymmetric Trend-Follow) ===
        desired_signal = 0.0
        
        # LONG: Weekly bull + Daily RSI pullback + 6h momentum + SMA confirm
        if weekly_bull and rsi_pullback_long and hma_6h_bull and sma_bull:
            desired_signal = SIZE
        
        # SHORT: Weekly bear + Daily RSI rally + 6h momentum + SMA confirm
        elif weekly_bear and rsi_pullback_short and hma_6h_bear and sma_bear:
            desired_signal = -SIZE
        
        # Fallback: Strong weekly trend with 6h confirm (relax daily RSI)
        elif weekly_bull and hma_6h_bull and sma_bull and daily_rsi < 60.0:
            desired_signal = SIZE * 0.6
        elif weekly_bear and hma_6h_bear and sma_bear and daily_rsi > 40.0:
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
            final_signal = SIZE * 0.6
        elif desired_signal <= -SIZE * 0.5:
            final_signal = -SIZE * 0.6
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