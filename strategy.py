#!/usr/bin/env python3
"""
Experiment #704: 12h Primary + 1d/1w HTF — HMA Trend + RSI Pullback + ATR Stop

Hypothesis: 12h timeframe reduces noise and fee drag while maintaining trend capture.
Using HMA(21/63) crossover for trend (63 = ~1 month on 12h), 1d/1w HMA for HTF bias,
and RSI(14) for entry timing with LOOSE conditions to ensure trade generation.

Key innovations:
1. 12h timeframe - proven to work best, 20-50 trades/year target
2. HMA(21/63) - 63 period = ~1 month on 12h, captures medium-term trend
3. 1d HMA(21) + 1w HMA(21) - dual HTF confirmation for direction bias
4. RSI(14) loose filter - <65 for long, >35 for short (NOT extreme values)
5. ATR(14) 2.5x trailing stop - risk management
6. Discrete sizing: 0.0, ±0.20, ±0.30 to minimize fee churn
7. Asymmetric logic - different entry thresholds for bull vs bear regimes

Entry conditions (LOOSE to ensure >=30 trades/year):
- LONG: 1d HMA bull + 1w HMA bull + HMA21>63 + RSI<65 (full size 0.30)
- LONG weak: 1d HMA bull + HMA21>63 + RSI<70 (half size 0.20)
- SHORT: 1d HMA bear + 1w HMA bear + HMA21<63 + RSI>35 (full size -0.30)
- SHORT weak: 1d HMA bear + HMA21<63 + RSI>30 (half size -0.20)

Target: Sharpe>0.40, trades>=30 train, trades>=3 test, DD>-40%
Timeframe: 12h
Size: 0.20-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_hma21_63_rsi_loose_1d1w_v1"
timeframe = "12h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - reduces lag while maintaining smoothness"""
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
    """Relative Strength Index - momentum oscillator"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.zeros(n)
    rsi[:] = np.nan
    for i in range(period, n):
        if avg_loss[i] > 1e-10:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
        else:
            rsi[i] = 100.0
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range - volatility measure for stops"""
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
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 12h indicators
    hma_21 = calculate_hma(close, period=21)
    hma_63 = calculate_hma(close, period=63)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.20
    SIZE_STRONG = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_21[i]) or np.isnan(hma_63[i]) or np.isnan(rsi[i]):
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
        
        # === HTF BIAS (1d and 1w HMA) ===
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        htf_1w_bull = close[i] > hma_1w_aligned[i]
        htf_1w_bear = close[i] < hma_1w_aligned[i]
        
        # === HMA CROSSOVER TREND ===
        hma_bull = hma_21[i] > hma_63[i]
        hma_bear = hma_21[i] < hma_63[i]
        
        # === RSI ENTRY (LOOSE - ensure trades) ===
        # Long on pullback in uptrend (RSI < 65, not extreme)
        rsi_long = rsi[i] < 65.0
        rsi_long_weak = rsi[i] < 70.0
        # Short on rally in downtrend (RSI > 35, not extreme)
        rsi_short = rsi[i] > 35.0
        rsi_short_weak = rsi[i] > 30.0
        
        # === ENTRY LOGIC (LOOSE CONDITIONS FOR TRADE GENERATION) ===
        desired_signal = 0.0
        
        # LONG: Full HTF alignment + HMA bull + RSI
        if htf_1d_bull and htf_1w_bull and hma_bull and rsi_long:
            desired_signal = SIZE_STRONG
        # LONG weak: 1d bias + HMA bull + RSI (looser)
        elif htf_1d_bull and hma_bull and rsi_long_weak:
            desired_signal = SIZE_BASE
        # LONG partial: HTF alignment + HMA bull (no RSI)
        elif htf_1d_bull and htf_1w_bull and hma_bull:
            desired_signal = SIZE_BASE
        
        # SHORT: Full HTF alignment + HMA bear + RSI
        elif htf_1d_bear and htf_1w_bear and hma_bear and rsi_short:
            desired_signal = -SIZE_STRONG
        # SHORT weak: 1d bias + HMA bear + RSI (looser)
        elif htf_1d_bear and hma_bear and rsi_short_weak:
            desired_signal = -SIZE_BASE
        # SHORT partial: HTF alignment + HMA bear (no RSI)
        elif htf_1d_bear and htf_1w_bear and hma_bear:
            desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (2.5x ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
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
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
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