#!/usr/bin/env python3
"""
Experiment #348: 4h Primary + 12h/1d HTF — Simplified Trend Pullback v1

Hypothesis: Regime-switching strategies (choppy vs trending) are too complex and
generate conflicting signals. Return to PROVEN trend-following with pullback entries.

Key insights from 347 failed experiments:
1. Regime switching (Choppiness Index) adds complexity without edge
2. Too many confluence filters = 0 trades (see #336, #337, #341, #345)
3. HMA crossovers alone get whipsawed in 2022 crash
4. Pullback entries in established trends have better R:R than breakouts

Strategy:
- 12h HMA = trend bias (only trade in direction)
- 1d HMA = regime confirmation (size 0.30 if aligned, 0.25 otherwise)
- 4h HMA(21) + RSI(14) pullback = entry trigger
- Entry: RSI pulls back to 40-55 zone while HMA trend intact
- Exit: ATR trailing stop (2.5x) or trend reversal

Why 4h works:
- 20-50 trades/year target (not too many, not too few)
- Enough bars for indicators to stabilize
- Captures multi-day trends without noise

Position sizing: 0.25 base, 0.30 when 1d HTF aligned (discrete levels)
Stoploss: 2.5x ATR(14) from entry, updated on new highs/lows

Target: Sharpe>0.40, DD>-40%, trades>=20 train, trades>=3 test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_trend_pullback_hma_rsi_12h1d_v1"
timeframe = "4h"
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
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF HMA for trend bias
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (4h) indicators
    hma_4h = calculate_hma(close, period=21)
    atr = calculate_atr(high, low, close, period=14)
    rsi = calculate_rsi(close, period=14)
    sma_50 = calculate_sma(close, 50)
    sma_200 = calculate_sma(close, 200)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_4h[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_12h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
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
        
        # === HTF TREND BIAS ===
        # 12h HMA direction
        htf_12h_bull = close[i] > hma_12h_aligned[i]
        htf_12h_bear = close[i] < hma_12h_aligned[i]
        
        # 1d HMA for size confirmation
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        # === 4h HMA TREND ===
        hma_bull = close[i] > hma_4h[i]
        hma_bear = close[i] < hma_4h[i]
        
        # === SMA200 LONG-TERM FILTER ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === RSI PULLBACK ZONES (LOOSENED for more trades) ===
        # Long: RSI pulled back to 40-55 in uptrend
        rsi_pullback_long = 38.0 <= rsi[i] <= 58.0
        # Short: RSI rallied to 42-62 in downtrend
        rsi_pullback_short = 42.0 <= rsi[i] <= 62.0
        
        # RSI momentum confirmation (not at extreme)
        rsi_not_extreme_long = rsi[i] < 70.0
        rsi_not_extreme_short = rsi[i] > 30.0
        
        # === ENTRY LOGIC (SIMPLIFIED - fewer confluence requirements) ===
        desired_signal = 0.0
        
        # LONG ENTRY: 12h bull + 4h HMA bull + RSI pullback + above SMA200
        if htf_12h_bull and hma_bull and rsi_pullback_long and rsi_not_extreme_long and above_sma200:
            # Size based on 1d alignment
            if htf_1d_bull:
                desired_signal = SIZE_STRONG
            else:
                desired_signal = SIZE_BASE
        
        # SHORT ENTRY: 12h bear + 4h HMA bear + RSI pullback + below SMA200
        elif htf_12h_bear and hma_bear and rsi_pullback_short and rsi_not_extreme_short and below_sma200:
            # Size based on 1d alignment
            if htf_1d_bear:
                desired_signal = -SIZE_STRONG
            else:
                desired_signal = -SIZE_BASE
        
        # === TRAILING STOPLOSS UPDATE ===
        if in_position and position_side > 0:
            # Update highest price since entry for long
            if close[i] > highest_since_entry:
                highest_since_entry = close[i]
            # Trail stop up
            new_stop = highest_since_entry - 2.5 * atr[i]
            if new_stop > stop_price:
                stop_price = new_stop
        
        if in_position and position_side < 0:
            # Update lowest price since entry for short
            if close[i] < lowest_since_entry:
                lowest_since_entry = close[i]
            # Trail stop down
            new_stop = lowest_since_entry + 2.5 * atr[i]
            if new_stop < stop_price:
                stop_price = new_stop
        
        # === STOPLOSS CHECK ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        # Exit long if 4h HMA turns bear
        if in_position and position_side > 0 and hma_bear:
            desired_signal = 0.0
        
        # Exit short if 4h HMA turns bull
        if in_position and position_side < 0 and hma_bull:
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
                highest_since_entry = close[i]
                lowest_since_entry = close[i]
                # Set initial stoploss
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
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = final_signal
    
    return signals