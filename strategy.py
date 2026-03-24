#!/usr/bin/env python3
"""
Experiment #1459: 4h Primary + 1d HTF — Simplified Dual Regime with Relaxed Thresholds

Hypothesis: Recent failures show 0 trades due to overly strict conditions. This strategy 
simplifies entry logic while keeping proven patterns (Choppiness regime + RSI + Donchian).

Key changes from failed experiments:
1. RELAXED thresholds: RSI 35/65 instead of 15/85, CHOP 55/45 instead of 61.8/38.2
2. OR logic for entries: multiple paths to trigger (mean revert OR breakout)
3. Simpler RSI(14) instead of complex CRSI (CRSI was too restrictive)
4. 1d HMA(21) macro filter only — don't over-complicate with multiple HTF

Why 4h not 1d:
- 4h gives 6x more opportunities than 1d while still filtering noise
- Target 30-60 trades/year on 4h (within fee drag limits)
- Research shows 4h HMA + RSI patterns work on SOL (Sharpe +0.879)

Design:
1. 1d HMA(21) = macro trend (call ONCE before loop via mtf_data)
2. 4h Choppiness(14) = regime (>55 range, <45 trend)
3. 4h RSI(14) = entry timing (oversold <35, overbought >65)
4. 4h Donchian(20) = breakout confirmation
5. Long: price>1d_HMA AND (chop>55+RSI<35 OR chop<45+Donchian breakout)
6. Short: price<1d_HMA AND (chop>55+RSI>65 OR chop<45+Donchian breakdown)
7. ATR(14) trailing stop 2.5x
8. Position size 0.28

Target: 40-80 trades over 4 years train, Sharpe > 0.618
Timeframe: 4h
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_chop_rsi_donchian_1d_hma_atr_relaxed_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(close, period=21):
    """Hull Moving Average - faster response than EMA"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def wma(series, span):
        weights = np.arange(1, span + 1, dtype=np.float64)
        result = np.full(len(series), np.nan)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1]
            if not np.any(np.isnan(window)):
                result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    hma = np.full(n, np.nan)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff = 2.0 * wma_half[i] - wma_full[i]
            if i >= sqrt_period - 1:
                diff_vals = []
                for j in range(i - sqrt_period + 1, i + 1):
                    if j >= period - 1 and not np.isnan(2.0 * wma_half[j] - wma_full[j]):
                        diff_vals.append(2.0 * wma_half[j] - wma_full[j])
                if len(diff_vals) == sqrt_period:
                    weights = np.arange(1, sqrt_period + 1, dtype=np.float64)
                    hma[i] = np.sum(np.array(diff_vals) * weights) / np.sum(weights)
    
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
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

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index - measures choppy vs trending"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    chop = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        highest_high = np.nanmax(high[i-period+1:i+1])
        lowest_low = np.nanmin(low[i-period+1:i+1])
        
        if highest_high > lowest_low:
            tr_sum = 0.0
            for j in range(i-period+1, i+1):
                if j > 0:
                    tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
                else:
                    tr = high[j] - low[j]
                tr_sum += tr
            
            chop[i] = 100.0 * np.log10(tr_sum / (highest_high - lowest_low)) / np.log10(period)
    
    return chop

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

def calculate_donchian(high, low, period=20):
    """Donchian Channel"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.nanmax(high[i-period+1:i+1])
        lower[i] = np.nanmin(low[i-period+1:i+1])
    
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d HMA for macro trend
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (4h) indicators
    donchian_20_upper, donchian_20_lower = calculate_donchian(high, low, period=20)
    rsi = calculate_rsi(close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.28
    
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
        if np.isnan(donchian_20_upper[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(rsi[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === MACRO TREND (1d HMA) ===
        macro_bull = close[i] > hma_1d_aligned[i]
        macro_bear = close[i] < hma_1d_aligned[i]
        
        # === CHOPPINESS REGIME (relaxed thresholds for more trades) ===
        is_choppy = chop[i] > 55.0
        is_trending = chop[i] < 45.0
        
        # === RSI EXTREMES (relaxed from 15/85 to 35/65) ===
        rsi_oversold = rsi[i] < 35.0
        rsi_overbought = rsi[i] > 65.0
        
        # === DONCHIAN BREAKOUT ===
        prev_upper = donchian_20_upper[i-1] if i > 0 else donchian_20_upper[i]
        prev_lower = donchian_20_lower[i-1] if i > 0 else donchian_20_lower[i]
        breakout_long = close[i] > prev_upper
        breakout_short = close[i] < prev_lower
        
        # === DESIRED SIGNAL - DUAL REGIME WITH OR LOGIC ===
        desired_signal = 0.0
        
        # LONG ENTRIES (multiple paths - any can trigger)
        # Path 1: Choppy + RSI oversold + macro bull (mean reversion)
        if is_choppy and rsi_oversold and macro_bull:
            desired_signal = BASE_SIZE
        # Path 2: Trending + Donchian breakout + macro bull
        elif is_trending and breakout_long and macro_bull:
            desired_signal = BASE_SIZE
        # Path 3: Neutral regime + RSI very oversold + macro bull (fallback)
        elif rsi[i] < 30.0 and macro_bull:
            desired_signal = BASE_SIZE * 0.5
        # Path 4: Strong breakout regardless of chop (momentum)
        elif breakout_long and macro_bull and rsi[i] > 50.0:
            desired_signal = BASE_SIZE
        
        # SHORT ENTRIES (multiple paths - any can trigger)
        # Path 1: Choppy + RSI overbought + macro bear (mean reversion)
        elif is_choppy and rsi_overbought and macro_bear:
            desired_signal = -BASE_SIZE
        # Path 2: Trending + Donchian breakdown + macro bear
        elif is_trending and breakout_short and macro_bear:
            desired_signal = -BASE_SIZE
        # Path 3: Neutral regime + RSI very overbought + macro bear (fallback)
        elif rsi[i] > 70.0 and macro_bear:
            desired_signal = -BASE_SIZE * 0.5
        # Path 4: Strong breakdown regardless of chop (momentum)
        elif breakout_short and macro_bear and rsi[i] < 50.0:
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
        if desired_signal >= BASE_SIZE * 0.4:
            final_signal = BASE_SIZE if desired_signal > 0 else -BASE_SIZE
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