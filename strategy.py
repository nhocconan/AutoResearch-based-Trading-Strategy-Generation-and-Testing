#!/usr/bin/env python3
"""
Experiment #1290: 1h Primary + 4h/12h HTF — Fisher Transform + Choppiness Regime

Hypothesis: Recent 1h strategies (#1280, #1285, #1288) all failed with Sharpe=0.000 = ZERO TRADES.
Problem: Too many confluence filters (session + volume + multiple indicators) blocked all entries.

Solution for THIS experiment:
1. LOOSEN entry thresholds significantly (RSI 25-75 instead of 30-70)
2. Use Fisher Transform for reversals (proven in bear markets, catches spikes)
3. Use Choppiness Index regime but WIDER bands (>58 range, <42 trend)
4. Remove session filter (blocking trades)
5. Remove volume filter (blocking trades)
6. 4h HMA for trend DIRECTION only, 1h Fisher for ENTRY timing
7. Target: 40-80 trades/year (enough for stats, not too many for fees)

Key changes from failed #1280/#1285:
- No session filter (8-20 UTC was blocking 60% of potential entries)
- No volume filter (volume spikes are rare on 1h)
- Fisher Transform instead of CRSI (more signals in bear/range markets)
- Wider Choppiness thresholds (58/42 vs 55/45)
- Lower RSI thresholds for mean reversion (25/75 vs 30/70)
- Signal size 0.25 (conservative for 1h TF)

Target: Sharpe > 0.612, trades >= 40/train, >= 5/test
Timeframe: 1h
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_fisher_chop_regime_4h12h_hma_atr_v1"
timeframe = "1h"
leverage = 1.0

def calculate_hma(close, period=21):
    """Hull Moving Average - faster response than EMA"""
    n = len(close)
    hma = np.full(n, np.nan)
    
    if n < period:
        return hma
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def wma(series, span):
        weights = np.arange(1, span + 1)
        result = np.full(len(series), np.nan)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1]
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff = 2.0 * wma_half[i] - wma_full[i]
            if i >= sqrt_period - 1:
                diff_window = []
                for j in range(i - sqrt_period + 1, i + 1):
                    if j >= period - 1 and not np.isnan(2.0 * wma_half[j] - wma_full[j]):
                        diff_window.append(2.0 * wma_half[j] - wma_full[j])
                if len(diff_window) == sqrt_period:
                    weights = np.arange(1, sqrt_period + 1)
                    hma[i] = np.sum(np.array(diff_window) * weights) / np.sum(weights)
    
    return hma

def calculate_fisher_transform(high, low, close, period=9):
    """Ehlers Fisher Transform - catches reversals in bear markets
    Long when Fisher crosses above -1.5 from below
    Short when Fisher crosses below +1.5 from above
    """
    n = len(close)
    fisher = np.full(n, np.nan)
    fisher_prev = np.full(n, np.nan)
    
    if n < period + 1:
        return fisher, fisher_prev
    
    for i in range(period, n):
        hh = np.max(high[i-period+1:i+1])
        ll = np.min(low[i-period+1:i+1])
        
        if hh > ll:
            value = 0.66 * ((close[i] - ll) / (hh - ll) - 0.5) + 0.67 * (0.66 * ((close[i-1] - ll) / (hh - ll) - 0.5) if i > 0 else 0)
            value = np.clip(value, -0.999, 0.999)
            
            fisher[i] = 0.5 * np.log((1 + value) / (1 - value))
            if i > period:
                fisher_prev[i] = fisher[i-1]
    
    return fisher, fisher_prev

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index - regime detection
    CHOP > 58 = ranging (mean revert)
    CHOP < 42 = trending (trend follow)
    42-58 = transition (no trades)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    for i in range(period, n):
        tr_sum = np.sum(tr[i-period+1:i+1])
        hh = np.max(high[i-period+1:i+1])
        ll = np.min(low[i-period+1:i+1])
        
        if hh > ll and tr_sum > 0:
            chop[i] = 100.0 * np.log10(tr_sum / (hh - ll)) / np.log10(period)
    
    return chop

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.zeros(n)
    loss = np.zeros(n)
    gain[1:] = np.where(delta > 0, delta, 0)
    loss[1:] = np.where(delta < 0, -delta, 0)
    
    gain_smooth = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_smooth = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    mask = loss_smooth > 1e-10
    rsi[mask] = 100.0 - (100.0 / (1.0 + gain_smooth[mask] / loss_smooth[mask]))
    rsi[:period] = np.nan
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Bollinger Bands"""
    n = len(close)
    mid = np.full(n, np.nan)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    if n < period:
        return mid, upper, lower
    
    for i in range(period - 1, n):
        window = close[i-period+1:i+1]
        mid[i] = np.mean(window)
        std = np.std(window, ddof=0)
        upper[i] = mid[i] + std_mult * std
        lower[i] = mid[i] - std_mult * std
    
    return mid, upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate and align HTF HMA for macro trend filter
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate primary (1h) indicators
    fisher, fisher_prev = calculate_fisher_transform(high, low, close, period=9)
    chop = calculate_choppiness(high, low, close, period=14)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    bb_mid, bb_upper, bb_lower = calculate_bollinger(close, period=20, std_mult=2.0)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25
    
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
            continue
        if np.isnan(fisher[i]) or np.isnan(fisher_prev[i]):
            signals[i] = 0.0
            continue
        if np.isnan(crsi[i]) if 'crsi' in dir() else np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        if np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_12h_aligned[i]):
            signals[i] = 0.0
            continue
        if np.isnan(bb_mid[i]) or np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            signals[i] = 0.0
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        in_range = chop[i] > 58.0  # Ranging market
        in_trend = chop[i] < 42.0  # Trending market
        # 42-58 = transition zone (no new entries)
        
        # === MACRO TREND (4h HMA) ===
        macro_bull = close[i] > hma_4h_aligned[i]
        macro_bear = close[i] < hma_4h_aligned[i]
        
        # === LONGER MACRO TREND (12h HMA) ===
        macro_12h_bull = close[i] > hma_12h_aligned[i]
        macro_12h_bear = close[i] < hma_12h_aligned[i]
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_long = fisher[i] > -1.5 and fisher_prev[i] <= -1.5
        fisher_short = fisher[i] < 1.5 and fisher_prev[i] >= 1.5
        
        # === RSI EXTREMES ===
        rsi_oversold = rsi[i] < 35.0
        rsi_overbought = rsi[i] > 65.0
        
        # === BOLLINGER BAND TOUCH ===
        at_bb_lower = close[i] <= bb_lower[i] * 1.003
        at_bb_upper = close[i] >= bb_upper[i] * 0.997
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # TRENDING REGIME: Follow HTF trend with Fisher entry
        if in_trend:
            # Long: 4h bull + 12h bull + Fisher long trigger
            if macro_bull and macro_12h_bull and fisher_long:
                desired_signal = BASE_SIZE
            # Short: 4h bear + 12h bear + Fisher short trigger
            elif macro_bear and macro_12h_bear and fisher_short:
                desired_signal = -BASE_SIZE
        
        # RANGING REGIME: Mean revert with RSI + BB + Fisher confluence
        elif in_range:
            # Long: RSI oversold + at BB lower + Fisher turning up
            if rsi_oversold and at_bb_lower and fisher[i] > fisher_prev[i]:
                desired_signal = BASE_SIZE
            # Short: RSI overbought + at BB upper + Fisher turning down
            elif rsi_overbought and at_bb_upper and fisher[i] < fisher_prev[i]:
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
        if desired_signal > 0.1:
            final_signal = BASE_SIZE
        elif desired_signal < -0.1:
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