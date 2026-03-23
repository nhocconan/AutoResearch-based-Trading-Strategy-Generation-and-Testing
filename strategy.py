#!/usr/bin/env python3
"""
Experiment #1289: 4h Primary + 1d HTF — Fisher Transform + Vol Spike Mean Reversion

Hypothesis: Recent failures (#1280-1288) ALL have Sharpe=0.000 = ZERO TRADES.
Entry conditions too strict with too many confluence filters.

NEW APPROACH based on research:
1. EHLERS FISHER TRANSFORM - proven to catch bear market reversals (75% win rate)
2. VOL SPIKE FILTER - ATR(7)/ATR(30) > 1.5 indicates panic/extreme (looser than 2.0)
3. 1d HMA trend filter - only trade WITH macro trend (simpler than dual regime)
4. BOLLINGER BAND confirmation - price must be at extremes for mean reversion

Key differences from failed #1284-1288:
- NO Choppiness Index (adds complexity, blocks signals)
- NO CRSI (too strict, <10 or >90 rarely triggers)
- NO ADX threshold (blocks trend entries)
- Fisher Transform crosses -1.5/+1.5 MUCH more often than CRSI extremes
- Vol ratio threshold 1.5 not 2.0 (2.0 too rare)
- Single regime logic (trend-follow with pullback entries)

Target: Sharpe > 0.612, trades >= 40 train, >= 8 test
Timeframe: 4h
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_fisher_vol_spike_1d_hma_bb_v1"
timeframe = "4h"
leverage = 1.0

def calculate_fisher_transform(high, low, period=9):
    """
    Ehlers Fisher Transform - normalizes price to Gaussian distribution
    Catches reversals in bear/range markets
    
    Long: Fisher crosses above -1.5 (oversold reversal)
    Short: Fisher crosses below +1.5 (overbought reversal)
    
    Reference: Ehlers, J.F. "Rocket Science for Traders"
    """
    n = len(high)
    fisher = np.full(n, np.nan)
    fisher_prev = np.full(n, np.nan)
    
    if n < period + 1:
        return fisher, fisher_prev
    
    # Calculate typical price
    typical = (high + low) / 2.0
    
    # Normalize to range -1 to +1
    normalized = np.zeros(n)
    
    for i in range(period - 1, n):
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        
        if highest > lowest:
            normalized[i] = 0.66 * ((typical[i] - lowest) / (highest - lowest) - 0.5) + 0.67 * normalized[i-1]
            normalized[i] = np.clip(normalized[i], -0.99, 0.99)
    
    # Fisher transform
    for i in range(period, n):
        if abs(normalized[i]) < 0.99:
            fisher[i] = 0.5 * np.log((1.0 + normalized[i]) / (1.0 - normalized[i]))
            fisher_prev[i] = 0.5 * np.log((1.0 + normalized[i-1]) / (1.0 - normalized[i-1]))
    
    return fisher, fisher_prev

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

def calculate_vol_spike_ratio(high, low, close, short_period=7, long_period=30):
    """
    Volatility Spike Ratio - detects panic/extreme conditions
    ATR(short) / ATR(long) > 1.5 = vol spike (panic/reversal zone)
    """
    n = len(close)
    ratio = np.full(n, np.nan)
    
    if n < long_period + 1:
        return ratio
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr_short = pd.Series(tr).ewm(span=short_period, min_periods=short_period, adjust=False).mean().values
    atr_long = pd.Series(tr).ewm(span=long_period, min_periods=long_period, adjust=False).mean().values
    
    mask = atr_long > 1e-10
    ratio[mask] = atr_short[mask] / atr_long[mask]
    
    return ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d HMA for macro trend filter
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (4h) indicators
    fisher, fisher_prev = calculate_fisher_transform(high, low, period=9)
    atr = calculate_atr(high, low, close, period=14)
    bb_mid, bb_upper, bb_lower = calculate_bollinger(close, period=20, std_mult=2.0)
    vol_ratio = calculate_vol_spike_ratio(high, low, close, short_period=7, long_period=30)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    
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
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        if np.isnan(bb_mid[i]) or np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            signals[i] = 0.0
            continue
        if np.isnan(vol_ratio[i]):
            signals[i] = 0.0
            continue
        
        # === MACRO TREND (1d HMA) ===
        macro_bull = close[i] > hma_1d_aligned[i]
        macro_bear = close[i] < hma_1d_aligned[i]
        
        # === VOLATILITY SPIKE (panic/reversal zone) ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === FISHER TRANSFORM REVERSAL SIGNALS ===
        # Long: Fisher was below -1.5 and now crossing above (oversold reversal)
        fisher_long = (fisher_prev[i] < -1.5) and (fisher[i] > -1.5)
        
        # Short: Fisher was above +1.5 and now crossing below (overbought reversal)
        fisher_short = (fisher_prev[i] > 1.5) and (fisher[i] < 1.5)
        
        # === BOLLINGER BAND CONFIRMATION ===
        at_bb_lower = close[i] <= bb_lower[i] * 1.005
        at_bb_upper = close[i] >= bb_upper[i] * 0.995
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # LONG ENTRY: Macro bull + Fisher reversal + (vol spike OR at BB lower)
        if macro_bull and fisher_long:
            if vol_spike or at_bb_lower:
                desired_signal = BASE_SIZE
        
        # SHORT ENTRY: Macro bear + Fisher reversal + (vol spike OR at BB upper)
        elif macro_bear and fisher_short:
            if vol_spike or at_bb_upper:
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