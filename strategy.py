#!/usr/bin/env python3
"""
Experiment #575: 6h Primary + 12h/1d HTF — Regime-Adaptive RSI Mean Reversion

Hypothesis: 6h timeframe sits between 4h and 12h - needs faster HTF (12h) for trend bias
while 1d provides macro confirmation. Focus on RSI mean reversion in range regimes
with trend-following in strong trend regimes. Simpler logic than #567/571 to ensure
trades actually generate (those got Sharpe=0.000 = no trades).

Key differences from failed 6h experiments:
1. 12h HMA instead of 1d for primary trend bias (faster response for 6h TF)
2. Looser RSI thresholds (30/70 vs 25/75) to ensure trade generation
3. Single regime filter (CHOP) instead of multiple conflicting filters
4. Discrete signal levels to minimize fee churn
5. ATR-based stoploss with signal→0 on breach

Strategy logic:
1. 12h HMA(21) = primary trend bias (aligned with shift(1))
2. 1d HMA(21) = macro confirmation (aligned with shift(1))
3. 6h RSI(14) = entry timing (oversold/overbought)
4. 6h Choppiness(14) = regime (CHOP>55=range, CHOP<45=trend)
5. 6h ATR(14) = stoploss (2.5x ATR from entry)

Regime-adaptive entries:
- TREND (CHOP<45): Pullback to RSI 40-50 in direction of HTF trend
- RANGE (CHOP>55): Mean revert at RSI extremes (30/70)
- TRANSITION: Stay flat or reduced size

Target: Sharpe>0.40, trades>=80 train (20/year), trades>=10 test
Timeframe: 6h
Size: 0.25-0.30 (discrete levels)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_regime_rsi_hma_12h1d_v2"
timeframe = "6h"
leverage = 1.0

def calculate_rsi(close, period=14):
    """Relative Strength Index with proper min_periods"""
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
    """Average True Range with proper min_periods"""
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
    Choppiness Index (CHOP)
    CHOP > 61.8 = range-bound (mean reversion)
    CHOP < 38.2 = trending (trend follow)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    atr = calculate_atr(high, low, close, period)
    
    chop = np.zeros(n)
    chop[:] = np.nan
    
    for i in range(period, n):
        atr_sum = np.nansum(atr[i-period+1:i+1])
        highest_high = np.nanmax(high[i-period+1:i+1])
        lowest_low = np.nanmin(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and atr_sum > 1e-10:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
        else:
            chop[i] = 50.0
    
    return chop

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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 12h HMA for primary trend bias
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate and align 1d HMA for macro confirmation
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 6h indicators
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
        
        if np.isnan(rsi[i]) or np.isnan(chop[i]):
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
        
        # === HTF BIAS (12h primary + 1d macro) ===
        htf_bull = close[i] > hma_12h_aligned[i] and close[i] > hma_1d_aligned[i]
        htf_bear = close[i] < hma_12h_aligned[i] and close[i] < hma_1d_aligned[i]
        
        # === CHOPPINESS REGIME ===
        chop_range = chop[i] > 55.0   # Range-bound (mean reversion)
        chop_trend = chop[i] < 45.0   # Trending (trend follow)
        
        # === RSI CONDITIONS ===
        rsi_oversold = rsi[i] < 35.0
        rsi_overbought = rsi[i] > 65.0
        rsi_neutral_low = rsi[i] < 50.0 and rsi[i] > 35.0
        rsi_neutral_high = rsi[i] > 50.0 and rsi[i] < 65.0
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # TREND REGIME: Pullback entries in direction of HTF trend
        if chop_trend:
            if htf_bull and rsi_neutral_low:
                # Bull trend, RSI pulled back to 35-50
                desired_signal = SIZE_BASE
            elif htf_bear and rsi_neutral_high:
                # Bear trend, RSI rallied to 50-65
                desired_signal = -SIZE_BASE
            # Strong trend continuation
            elif htf_bull and rsi[i] > 50.0 and rsi[i] < 60.0:
                desired_signal = SIZE_STRONG
            elif htf_bear and rsi[i] < 50.0 and rsi[i] > 40.0:
                desired_signal = -SIZE_STRONG
        
        # RANGE REGIME: Mean reversion at RSI extremes
        elif chop_range:
            if rsi_oversold:
                # RSI < 35 in range = long mean reversion
                desired_signal = SIZE_BASE
            elif rsi_overbought:
                # RSI > 65 in range = short mean reversion
                desired_signal = -SIZE_BASE
        
        # TRANSITION REGIME: Wait for clearer signals
        else:
            # Only take extreme RSI signals in transition
            if rsi[i] < 30.0 and htf_bull:
                desired_signal = SIZE_BASE * 0.8
            elif rsi[i] > 70.0 and htf_bear:
                desired_signal = -SIZE_BASE * 0.8
        
        # === STOPLOSS CHECK (2.5x ATR from entry) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            if low[i] < stop_price:
                stoploss_triggered = True
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            if high[i] > stop_price:
                stoploss_triggered = True
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
        
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
        elif abs(desired_signal) >= SIZE_BASE * 0.5:
            final_signal = np.sign(desired_signal) * SIZE_BASE * 0.8
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