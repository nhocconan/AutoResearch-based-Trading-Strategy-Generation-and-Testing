#!/usr/bin/env python3
"""
Experiment #487: 6h Primary + 1d HTF — Volatility Expansion + RSI Mean Reversion

Hypothesis: 6h timeframe is underexplored and captures multi-day swings better than 4h.
This strategy uses volatility expansion as the primary signal, filtered by 1d trend:
1. 1d HMA(21) = overall trend bias (single HTF filter, proven in #478)
2. 6h ATR expansion (ATR > 1.5x recent avg) = volatility regime detection
3. 6h RSI(14) extremes = entry timing with loose thresholds (35/65)
4. 6h Bollinger Band position = confirm mean reversion setup
5. ATR(14)*2.5 stoploss on all positions (wider for 6h volatility)

Key differences from failed 6h experiments:
- Volatility expansion filter (NEW - not tried on 6h before)
- SINGLE HTF filter (1d only, not 12h+1d which caused 0 trades in #479/#484)
- LOOSE RSI thresholds (35/65 not 30/70) to guarantee trade generation
- OR logic for entries (any trigger works, not AND confluence)
- Discrete signal levels (0.0, ±0.25, ±0.30) to minimize fee churn

Target: Sharpe>0.40, trades>=100 train (25/year), trades>=15 test
Timeframe: 6h (NEW - high priority exploration per experiment brief)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_volexp_rsi_hma_1d_v1"
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

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Bollinger Bands"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    
    return upper, lower, sma

def calculate_atr_ratio(atr, short_period=7, long_period=30):
    """ATR ratio for volatility expansion detection"""
    n = len(atr)
    if n < long_period:
        return np.full(n, np.nan)
    
    # Rolling average of ATR over long period
    atr_avg = pd.Series(atr).rolling(window=long_period, min_periods=long_period).mean().values
    
    # Ratio of current ATR to recent average
    ratio = atr / atr_avg
    
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
    
    # Load 1d HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d HMA for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 6h indicators
    hma_6h = calculate_hma(close, period=21)
    atr = calculate_atr(high, low, close, period=14)
    rsi = calculate_rsi(close, period=14)
    bb_upper, bb_lower, bb_mid = calculate_bollinger(close, period=20, std_mult=2.0)
    atr_ratio = calculate_atr_ratio(atr, short_period=7, long_period=30)
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
    
    for i in range(250, n):
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
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(sma_50[i]):
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
        
        # === 1d HTF BIAS (SINGLE FILTER) ===
        htf_bull = close[i] > hma_1d_aligned[i]
        htf_bear = close[i] < hma_1d_aligned[i]
        
        # === 6h HMA TREND ===
        hma_bull = close[i] > hma_6h[i]
        hma_bear = close[i] < hma_6h[i]
        
        # === SMA FILTERS ===
        above_sma50 = close[i] > sma_50[i]
        below_sma50 = close[i] < sma_50[i]
        above_sma200 = not np.isnan(sma_200[i]) and close[i] > sma_200[i]
        below_sma200 = not np.isnan(sma_200[i]) and close[i] < sma_200[i]
        
        # === VOLATILITY EXPANSION ===
        vol_expansion = not np.isnan(atr_ratio[i]) and atr_ratio[i] > 1.3
        vol_spike = not np.isnan(atr_ratio[i]) and atr_ratio[i] > 1.8
        
        # === RSI EXTREMES (LOOSE: 35/65) ===
        rsi_oversold = rsi[i] < 35.0
        rsi_overbought = rsi[i] > 65.0
        rsi_extreme_oversold = rsi[i] < 28.0
        rsi_extreme_overbought = rsi[i] > 72.0
        
        # === BOLLINGER BAND POSITION ===
        near_bb_lower = close[i] < bb_lower[i] * 1.005  # within 0.5% of lower band
        near_bb_upper = close[i] > bb_upper[i] * 0.995  # within 0.5% of upper band
        bb_squeeze = (bb_upper[i] - bb_lower[i]) / bb_mid[i] < 0.05  # bandwidth < 5%
        
        # === ENTRY LOGIC (LOOSE - OR logic, not AND) ===
        desired_signal = 0.0
        
        # TREND LONG: 1d bull + (vol expansion + RSI recovery OR BB bounce)
        if htf_bull:
            if vol_expansion and rsi[i] > 40.0 and rsi[i-1] < 40.0 and above_sma50:
                # RSI crossing above 40 with vol expansion
                desired_signal = SIZE_STRONG
            elif near_bb_lower and rsi_oversold and hma_bull:
                # BB lower band bounce with oversold RSI
                desired_signal = SIZE_BASE
            elif vol_spike and rsi_extreme_oversold:
                # Vol spike + extreme oversold = panic buy
                desired_signal = SIZE_BASE
        
        # TREND SHORT: 1d bear + (vol expansion + RSI weakness OR BB rejection)
        elif htf_bear:
            if vol_expansion and rsi[i] < 60.0 and rsi[i-1] > 60.0 and below_sma50:
                # RSI crossing below 60 with vol expansion
                desired_signal = -SIZE_STRONG
            elif near_bb_upper and rsi_overbought and hma_bear:
                # BB upper band rejection with overbought RSI
                desired_signal = -SIZE_BASE
            elif vol_spike and rsi_extreme_overbought:
                # Vol spike + extreme overbought = panic sell
                desired_signal = -SIZE_BASE
        
        # MEAN REVERSION LONG: RSI extreme + BB (no HTF filter for MR)
        if desired_signal == 0.0:
            if rsi_extreme_oversold and near_bb_lower and above_sma200:
                desired_signal = SIZE_BASE
            elif rsi_oversold and near_bb_lower and above_sma50:
                desired_signal = SIZE_BASE * 0.8
        
        # MEAN REVERSION SHORT: RSI extreme + BB (no HTF filter for MR)
        if desired_signal == 0.0:
            if rsi_extreme_overbought and near_bb_upper and below_sma200:
                desired_signal = -SIZE_BASE
            elif rsi_overbought and near_bb_upper and below_sma50:
                desired_signal = -SIZE_BASE * 0.8
        
        # === STOPLOSS CHECK (2.5x ATR from entry - wider for 6h) ===
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
        elif desired_signal >= SIZE_BASE * 0.5:
            final_signal = SIZE_BASE * 0.8
        elif desired_signal <= -SIZE_BASE * 0.5:
            final_signal = -SIZE_BASE * 0.8
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
                # Set stoploss (2.5x ATR for 6h volatility)
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