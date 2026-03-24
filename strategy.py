#!/usr/bin/env python3
"""
Experiment #202: 4h Primary + 1d/1w HTF — HMA Trend + RSI Pullback + Regime Filter

Hypothesis: 4h timeframe strikes the right balance between trade frequency (20-50/year)
and signal quality. Previous 4h attempt (#198) failed with Sharpe=-0.292, likely due to
too many conflicting filters. This version SIMPLIFIES entry logic:

Core Logic:
- 1d HMA(50) = major trend bias (only trade in direction)
- 4h HMA(21) vs HMA(50) = intermediate trend (crossover signal)
- RSI(14) pullback = entry timing (buy dips in uptrend, sell rallies in downtrend)
- Choppiness Index = regime filter (avoid mean-reversion entries in strong trends)
- ATR(14) trailing stop = risk management (2.5x ATR)

Key Changes from #198:
- Fewer filters (removed ADX, removed volume filters that killed trades)
- RSI thresholds widened (30-70 instead of 25-75) to generate more trades
- HMA crossover as primary signal, RSI as confirmation (not reverse)
- 1w HMA for ultra-long-term bias (avoid counter-trend trades)

Position sizing: 0.25 base, 0.30 strong signals
Target: Sharpe>0.40 (beat current best 0.399), trades>=30 train, trades>=3 test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_hma_rsi_pullback_1d1w_v3"
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

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures market choppiness vs trending
    CHOP > 61.8 = choppy/range bound
    CHOP < 38.2 = trending
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.zeros(n)
    chop[:] = np.nan
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop

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
    
    # Calculate and align 1w HMA for ultra-long-term bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=20)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (4h) indicators
    hma_4h_fast = calculate_hma(close, period=21)
    hma_4h_slow = calculate_hma(close, period=50)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
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
        if np.isnan(hma_4h_fast[i]) or np.isnan(hma_4h_slow[i]):
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
        if np.isnan(rsi[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1d and 1w HMA) ===
        price_vs_1d = close[i] / hma_1d_aligned[i] if hma_1d_aligned[i] > 0 else 1.0
        price_vs_1w = close[i] / hma_1w_aligned[i] if hma_1w_aligned[i] > 0 else 1.0
        
        htf_bull = price_vs_1d > 1.02 and price_vs_1w > 1.0  # 2% above 1d HMA
        htf_bear = price_vs_1d < 0.98 and price_vs_1w < 1.0  # 2% below 1d HMA
        htf_neutral = not htf_bull and not htf_bear
        
        # === 4h HMA CROSSOVER TREND ===
        hma_cross_bull = hma_4h_fast[i] > hma_4h_slow[i]
        hma_cross_bear = hma_4h_fast[i] < hma_4h_slow[i]
        
        # === RSI PULLBACK CONDITIONS ===
        rsi_oversold = rsi[i] < 45.0  # Pullback in uptrend
        rsi_overbought = rsi[i] > 55.0  # Rally in downtrend
        rsi_extreme_low = rsi[i] < 35.0  # Strong oversold
        rsi_extreme_high = rsi[i] > 65.0  # Strong overbought
        
        # === CHOPPINESS REGIME ===
        is_trending = chop[i] < 50.0  # Below 50 = more trending
        is_choppy = chop[i] > 55.0  # Above 55 = choppy
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # LONG entries
        if htf_bull or htf_neutral:
            # Primary: HMA cross bull + RSI pullback (not extreme)
            if hma_cross_bull and rsi_oversold and not rsi_extreme_low:
                if is_trending:
                    desired_signal = SIZE_STRONG
                else:
                    desired_signal = SIZE_BASE
            
            # Secondary: HMA cross bull + RSI extreme (strong signal even in chop)
            elif hma_cross_bull and rsi_extreme_low:
                desired_signal = SIZE_STRONG
            
            # Tertiary: Price above both HMA + RSI recovering from oversold
            elif (close[i] > hma_4h_fast[i] and close[i] > hma_4h_slow[i] 
                  and rsi[i] > 40 and rsi[i] < 55):
                if htf_bull:
                    desired_signal = SIZE_BASE
        
        # SHORT entries
        elif htf_bear or htf_neutral:
            # Primary: HMA cross bear + RSI rally (not extreme)
            if hma_cross_bear and rsi_overbought and not rsi_extreme_high:
                if is_trending:
                    desired_signal = -SIZE_STRONG
                else:
                    desired_signal = -SIZE_BASE
            
            # Secondary: HMA cross bear + RSI extreme (strong signal even in chop)
            elif hma_cross_bear and rsi_extreme_high:
                desired_signal = -SIZE_STRONG
            
            # Tertiary: Price below both HMA + RSI declining from overbought
            elif (close[i] < hma_4h_fast[i] and close[i] < hma_4h_slow[i]
                  and rsi[i] < 60 and rsi[i] > 45):
                if htf_bear:
                    desired_signal = -SIZE_BASE
        
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