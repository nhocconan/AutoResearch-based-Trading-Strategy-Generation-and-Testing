#!/usr/bin/env python3
"""
Experiment #1602: 12h Primary + 1d/1w HTF — Dual Regime Strategy with Choppiness Index

Hypothesis: After analyzing 11 failed 4h experiments and reviewing #1596 (12h HMA crossover got Sharpe=0.408),
the 12h timeframe shows promise. This strategy uses a DUAL REGIME approach:

Key innovations:
1. CHOPPINESS INDEX (14) regime detection - switch between trend-follow and mean-revert
   - CHOP > 61.8 = ranging market → use RSI mean reversion at Bollinger bands
   - CHOP < 38.2 = trending market → use HMA/Donchian breakout
   - 38.2 < CHOP < 61.8 = transition → stay flat or reduce position
2. 1d HMA(21) for trend bias - only trade with daily trend direction
3. 1w HMA(21) for long-term regime filter - avoid counter-weekly-trend trades
4. RSI(14) extremes for mean-revert entries (<30 long, >70 short)
5. Donchian(20) breakout for trend-follow entries
6. ATR(14) 2.5x trailing stop for drawdown control
7. Discrete position sizing (0.25) to minimize fee churn

Why this should beat Sharpe 0.618:
- Dual regime adapts to market conditions (trend vs range) - proven in academic literature
- 12h timeframe = 20-50 trades/year optimal for fee efficiency
- CHOP filter prevents trend-follow whipsaws in choppy markets (major issue in 2022-2025)
- Mean-revert component captures range-bound periods (2025 bear/range market)
- Dual HTF filter (1d + 1w) ensures we trade with major trend
- LOOSE entry conditions to guarantee ≥10 trades/symbol

Timeframe: 12h (required for this experiment)
HTF: 1d HMA + 1w HMA for bias (use mtf_data helper - call ONCE before loop)
Target: Sharpe > 0.618, trades > 10/symbol train, > 3/symbol test, DD > -50%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_chop_regime_dual_hma_1d1w_rsi_donchian_atr_v1"
timeframe = "12h"
leverage = 1.0

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP) - measures market choppiness vs trending
    Formula: 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    CHOP > 61.8 = ranging market
    CHOP < 38.2 = trending market
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    # Calculate True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.full(n, np.nan)
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and atr_sum > 1e-10:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
        else:
            chop[i] = 50.0  # neutral
    
    return chop

def calculate_hma(close, period=21):
    """
    Hull Moving Average - reduces lag while maintaining smoothness
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = max(period // 2, 1)
    sqrt_period = max(int(np.sqrt(period)), 1)
    
    def wma(data, w_period):
        if w_period < 1:
            return np.full(len(data), np.nan)
        result = np.full(len(data), np.nan)
        weights = np.arange(1, w_period + 1, dtype=float)
        weights /= weights.sum()
        for i in range(w_period - 1, len(data)):
            window = data[i - w_period + 1:i + 1]
            if np.any(np.isnan(window)):
                continue
            result[i] = np.sum(window * weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan)
    mask = ~np.isnan(wma_half) & ~np.isnan(wma_full)
    diff[mask] = 2.0 * wma_half[mask] - wma_full[mask]
    
    hma = wma(diff, sqrt_period)
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index with proper min_periods"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    gain_smooth = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_smooth = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan)
    mask = loss_smooth > 1e-10
    rsi[mask] = 100.0 - (100.0 / (1.0 + gain_smooth[mask] / loss_smooth[mask]))
    rsi[loss_smooth <= 1e-10] = 100.0
    rsi[:period] = np.nan
    
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

def calculate_donchian(high, low, period=20):
    """Donchian Channel - highest high and lowest low over period"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i-period+1:i+1])
        lower[i] = np.min(low[i-period+1:i+1])
    
    return upper, lower

def calculate_bollinger(close, period=20, std_dev=2.0):
    """Bollinger Bands with proper min_periods"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    sma = np.full(n, np.nan)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        sma[i] = np.mean(close[i-period+1:i+1])
        std = np.std(close[i-period+1:i+1])
        upper[i] = sma[i] + std_dev * std
        lower[i] = sma[i] - std_dev * std
    
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1d HMA for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate and align 1w HMA for long-term regime
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (12h) indicators
    atr = calculate_atr(high, low, close, period=14)
    rsi = calculate_rsi(close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    
    # Donchian channels for breakout
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
    # Bollinger Bands for mean reversion
    bb_upper, bb_lower = calculate_bollinger(close, period=20, std_dev=2.0)
    
    # HMA for trend following
    hma_fast = calculate_hma(close, period=16)
    hma_slow = calculate_hma(close, period=48)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
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
        if np.isnan(chop[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_ranging = chop[i] > 61.8
        is_trending = chop[i] < 38.2
        is_transition = not is_ranging and not is_trending
        
        # === TREND BIAS (1d HMA + 1w HMA) ===
        daily_bull = close[i] > hma_1d_aligned[i]
        daily_bear = close[i] < hma_1d_aligned[i]
        weekly_bull = close[i] > hma_1w_aligned[i]
        weekly_bear = close[i] < hma_1w_aligned[i]
        
        # === HMA TREND SIGNAL ===
        hma_bull = not np.isnan(hma_fast[i]) and not np.isnan(hma_slow[i]) and hma_fast[i] > hma_slow[i]
        hma_bear = not np.isnan(hma_fast[i]) and not np.isnan(hma_slow[i]) and hma_fast[i] < hma_slow[i]
        
        # === DONCHIAN BREAKOUT ===
        donchian_bull = not np.isnan(donchian_upper[i]) and close[i] >= donchian_upper[i]
        donchian_bear = not np.isnan(donchian_lower[i]) and close[i] <= donchian_lower[i]
        
        # === RSI EXTREMES (for mean reversion) ===
        rsi_oversold = rsi[i] < 30.0
        rsi_overbought = rsi[i] > 70.0
        
        # === BOLLINGER BAND POSITION ===
        at_bb_lower = not np.isnan(bb_lower[i]) and close[i] <= bb_lower[i]
        at_bb_upper = not np.isnan(bb_upper[i]) and close[i] >= bb_upper[i]
        
        # === PRIMARY SIGNAL (Dual Regime) ===
        desired_signal = 0.0
        
        if is_trending:
            # TREND-FOLLOWING MODE
            # Long: HMA bull + Daily bull + Weekly aligned + Donchian breakout
            if hma_bull and daily_bull and (weekly_bull or not weekly_bear) and donchian_bull:
                desired_signal = BASE_SIZE
            
            # Short: HMA bear + Daily bear + Weekly aligned + Donchian breakout
            elif hma_bear and daily_bear and (weekly_bear or not weekly_bull) and donchian_bear:
                desired_signal = -BASE_SIZE
        
        elif is_ranging:
            # MEAN REVERSION MODE
            # Long: RSI oversold + at BB lower + Daily bull (trade with trend)
            if rsi_oversold and at_bb_lower and daily_bull:
                desired_signal = BASE_SIZE
            
            # Short: RSI overbought + at BB upper + Daily bear (trade with trend)
            elif rsi_overbought and at_bb_upper and daily_bear:
                desired_signal = -BASE_SIZE
        
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
        if desired_signal >= BASE_SIZE * 0.85:
            final_signal = BASE_SIZE
        elif desired_signal <= -BASE_SIZE * 0.85:
            final_signal = -BASE_SIZE
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