#!/usr/bin/env python3
"""
Experiment #891: 6h Primary + 1w/1d HTF — Donchian Breakout + HMA Trend + Regime Filter

Hypothesis: 6h Donchian(20) breakouts capture momentum moves while 1w/1d HMA filters
prevent false breakouts against the major trend. Choppiness Index avoids breakout
trades in ranging markets (where breakouts fail 70%+ of time). This combines proven
breakout mechanics with multi-timeframe trend confirmation.

Key innovations:
1. 1w HMA(21) for ultra-long-term bias (changes ~4x/year)
2. 1d HMA(21) for medium-term trend (changes ~12x/year)
3. 6h Donchian(20) breakout for entry timing
4. Choppiness Index(14) < 50 = only take breakouts (trending regime)
5. Choppiness Index(14) >= 50 = mean revert at Donchian bounds
6. ATR(14) 2.5x trailing stop for risk management
7. Discrete sizing: 0.0, ±0.20, ±0.30

Entry conditions (LOOSE to ensure ≥10 trades/train, ≥3/test):
- TREND BREAKOUT: 1w+1d HMA bull + price breaks Donchian high + CHOP<50
- TREND BREAKOUT: 1w+1d HMA bear + price breaks Donchian low + CHOP<50
- RANGE REVERT: 1d HMA bull + price at Donchian low + CHOP>=50 (long)
- RANGE REVERT: 1d HMA bear + price at Donchian high + CHOP>=50 (short)

Target: Sharpe>0.45, trades>=30 train, trades>=5 test, DD>-40%
Timeframe: 6h
Size: 0.20-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_donchian_hma_chop_regime_1d1w_v1"
timeframe = "6h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - reduces lag while maintaining smoothness"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_n = int(np.sqrt(period))
    
    def wma(series, span):
        result = np.full(len(series), np.nan)
        weights = np.arange(1, span + 1, dtype=np.float64)
        weight_sum = np.sum(weights)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1]
            result[i] = np.sum(window * weights) / weight_sum
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff[i] = 2 * wma_half[i] - wma_full[i]
    
    hma = wma(diff, sqrt_n)
    return hma

def calculate_donchian(high, low, period=20):
    """Donchian Channel - highest high and lowest low over period"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower

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
    """Choppiness Index - detects trending vs ranging markets"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.full(n, np.nan)
    
    for i in range(period, n):
        sum_tr = np.sum(tr[i - period + 1:i + 1])
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and sum_tr > 1e-10:
            chop[i] = 100.0 * np.log10(sum_tr / price_range) / np.log10(period)
        else:
            chop[i] = 50.0
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (CRITICAL - Rule 1)
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF HMA
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 6h indicators
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    atr_14 = calculate_atr(high, low, close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    
    # Donchian middle
    donchian_mid = np.full(n, np.nan)
    for i in range(20, n):
        if not np.isnan(donchian_upper[i]) and not np.isnan(donchian_lower[i]):
            donchian_mid[i] = (donchian_upper[i] + donchian_lower[i]) / 2.0
    
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
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(chop_14[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1w_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1w and 1d HMA) ===
        htf_1w_bull = close[i] > hma_1w_aligned[i]
        htf_1w_bear = close[i] < hma_1w_aligned[i]
        
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        # === DONCHIAN BREAKOUT DETECTION ===
        # Price breaking above upper band
        breakout_long = False
        if i > 0 and not np.isnan(donchian_upper[i-1]):
            breakout_long = (close[i-1] <= donchian_upper[i-1]) and (close[i] > donchian_upper[i])
        
        # Price breaking below lower band
        breakout_short = False
        if i > 0 and not np.isnan(donchian_lower[i-1]):
            breakout_short = (close[i-1] >= donchian_lower[i-1]) and (close[i] < donchian_lower[i])
        
        # Price at Donchian bounds (for mean reversion)
        at_lower_bound = close[i] <= donchian_lower[i] * 1.002  # Within 0.2% of lower
        at_upper_bound = close[i] >= donchian_upper[i] * 0.998  # Within 0.2% of upper
        
        # === CHOPPINESS REGIME ===
        chop_trending = chop_14[i] < 50.0
        chop_ranging = chop_14[i] >= 50.0
        
        # === ENTRY LOGIC (REGIME ADAPTIVE + LOOSE FOR TRADES) ===
        desired_signal = 0.0
        
        # Strong HTF agreement (both 1w and 1d same direction)
        htf_strong_bull = htf_1w_bull and htf_1d_bull
        htf_strong_bear = htf_1w_bear and htf_1d_bear
        
        # Weak HTF (only 1d agrees)
        htf_weak_bull = htf_1d_bull and not htf_1w_bull
        htf_weak_bear = htf_1d_bear and not htf_1w_bear
        
        if htf_strong_bull:
            # Strong bullish bias - look for long entries
            if chop_trending:
                # Trend regime: take breakout longs
                if breakout_long:
                    desired_signal = SIZE_STRONG
                elif close[i] > donchian_mid[i] and close[i-1] <= donchian_mid[i-1]:
                    # Cross above middle
                    desired_signal = SIZE_BASE
            else:
                # Range regime: mean revert at lower bound
                if at_lower_bound:
                    desired_signal = SIZE_BASE
        
        elif htf_strong_bear:
            # Strong bearish bias - look for short entries
            if chop_trending:
                # Trend regime: take breakout shorts
                if breakout_short:
                    desired_signal = -SIZE_STRONG
                elif close[i] < donchian_mid[i] and close[i-1] >= donchian_mid[i-1]:
                    # Cross below middle
                    desired_signal = -SIZE_BASE
            else:
                # Range regime: mean revert at upper bound
                if at_upper_bound:
                    desired_signal = -SIZE_BASE
        
        elif htf_weak_bull:
            # Weak bullish bias (only 1d)
            if chop_trending and breakout_long:
                desired_signal = SIZE_BASE * 0.75
            elif chop_ranging and at_lower_bound:
                desired_signal = SIZE_BASE * 0.5
        
        elif htf_weak_bear:
            # Weak bearish bias (only 1d)
            if chop_trending and breakout_short:
                desired_signal = -SIZE_BASE * 0.75
            elif chop_ranging and at_upper_bound:
                desired_signal = -SIZE_BASE * 0.5
        
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
        elif desired_signal >= SIZE_BASE * 0.5:
            final_signal = SIZE_BASE * 0.5
        elif desired_signal <= -SIZE_BASE * 0.5:
            final_signal = -SIZE_BASE * 0.5
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
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