#!/usr/bin/env python3
"""
Experiment #344: 4h Primary + 12h/1d HTF — Donchian Breakout with Regime Switch

Hypothesis: Previous 4h strategy #339 failed (Sharpe=-0.458) due to:
1. Too many confluence filters (1d KAMA + Chop + Volume + RSI + KAMA crossover)
2. Complex hold logic causing signal churn
3. Asymmetric bias too restrictive for range markets

This strategy uses:
1. 12h HMA(21) for MACRO TREND DIRECTION (simpler, faster than 1d KAMA)
2. 4h Donchian(20) breakout for trend entries (proven on SOL)
3. 4h Choppiness Index to switch between trend/mean-revert modes
4. 4h RSI(14) for pullback entries in established trends
5. Simple ATR trailing stop (2.5x) — no complex hold logic

KEY INSIGHT: In trending regime (CHOP<45), trade Donchian breakouts with trend.
In choppy regime (CHOP>55), trade RSI extremes for mean reversion.
This dual-mode approach adapts to market conditions without over-filtering.

TARGET: 30-50 trades/year on 4h, Sharpe > 0.6 on ALL symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_regime_12h_hma_chop_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(close, period):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    wma_half = close_s.rolling(window=period//2, min_periods=period//2).mean()
    wma_full = close_s.rolling(window=period, min_periods=period).mean()
    raw_hma = 2 * wma_half - wma_full
    hma = raw_hma.rolling(window=int(np.sqrt(period)), min_periods=int(np.sqrt(period))).mean()
    return hma.values

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_choppiness(high, low, close, period=14):
    """Calculate Choppiness Index."""
    atr = calculate_atr(high, low, close, period)
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    sum_atr = pd.Series(atr).rolling(window=period, min_periods=period).sum().values
    with np.errstate(divide='ignore', invalid='ignore'):
        chop = 100 * np.log10(sum_atr / (highest_high - lowest_low + 1e-10)) / np.log10(period)
    chop = np.nan_to_num(chop, nan=50.0)
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0).values

def calculate_donchian_channels(high, low, period=20):
    """Calculate Donchian Channel upper and lower bands."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 4h indicators (primary timeframe)
    atr_14 = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    donchian_upper, donchian_lower = calculate_donchian_channels(high, low, period=20)
    
    # Calculate and align 12h HMA for macro bias
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.28  # 28% position size for 4h
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        if np.isnan(chop[i]) or np.isnan(rsi_14[i]):
            signals[i] = 0.0
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            continue
        if np.isnan(hma_12h_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === MACRO BIAS (12h HMA) ===
        price_above_hma_12h = close[i] > hma_12h_aligned[i]
        price_below_hma_12h = close[i] < hma_12h_aligned[i]
        
        # === REGIME DETECTION ===
        is_trending = chop[i] < 45.0  # Low choppiness = trend
        is_choppy = chop[i] > 55.0    # High choppiness = range
        
        # === TREND BREAKOUT SIGNALS ===
        breakout_long = close[i] > donchian_upper[i-1]  # Break above previous high
        breakout_short = close[i] < donchian_lower[i-1]  # Break below previous low
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        if is_trending:
            # TREND REGIME: Trade Donchian breakouts with 12h bias
            if price_above_hma_12h and breakout_long:
                desired_signal = BASE_SIZE
            elif price_below_hma_12h and breakout_short:
                desired_signal = -BASE_SIZE
            # Pullback entries in established trend
            elif price_above_hma_12h and 35 <= rsi_14[i] <= 50:
                desired_signal = BASE_SIZE * 0.7
            elif price_below_hma_12h and 50 <= rsi_14[i] <= 65:
                desired_signal = -BASE_SIZE * 0.7
        
        elif is_choppy:
            # RANGE REGIME: Mean reversion at extremes
            if rsi_14[i] < 28:
                desired_signal = BASE_SIZE * 0.6
            elif rsi_14[i] > 72:
                desired_signal = -BASE_SIZE * 0.6
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
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
        
        # === TAKE PROFIT (2.5R target) ===
        take_profit_triggered = False
        
        if in_position and position_side > 0:
            profit = close[i] - entry_price
            if profit >= 2.5 * entry_atr:
                take_profit_triggered = True
        
        if in_position and position_side < 0:
            profit = entry_price - close[i]
            if profit >= 2.5 * entry_atr:
                take_profit_triggered = True
        
        if take_profit_triggered:
            desired_signal = 0.0
        
        # === RSI EXTREME EXIT ===
        if in_position and position_side > 0 and rsi_14[i] > 78:
            desired_signal = 0.0
        if in_position and position_side < 0 and rsi_14[i] < 22:
            desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Position flip
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = desired_signal
    
    return signals