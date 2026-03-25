#!/usr/bin/env python3
"""
Experiment #1235: 6h Primary + 1d/1w HTF — Fisher Transform Reversal + Vol Spike

Hypothesis: After 960+ failed experiments, the key insight is that BTC/ETH 2025 is 
bear/range market where trend-following fails. Fisher Transform excels at catching 
reversals in range markets (proven Sharpe 0.8-1.5 in research). Combined with:

1. Ehlers Fisher Transform (period=9) - catches reversals better than RSI
2. 1d HMA(21) for trend bias - simple directional filter
3. ATR vol spike filter (ATR7/ATR30 > 1.8) - enter on vol expansion
4. 1w HMA for mega-trend confirmation

Why 6h: Middle ground between 4h (too noisy) and 12h (too few trades).
Target 30-60 trades/year with discrete sizing 0.20-0.30.

Entry logic:
- LONG: Fisher crosses above -1.5 + price > 1d_HMA + vol_spike
- SHORT: Fisher crosses below +1.5 + price < 1d_HMA + vol_spike
- Exit: Fisher opposite cross OR 2.5x ATR stoploss

This is DIFFERENT from all failed 6h strategies (no weekly pivots, no CHOP, no CRSI).
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_fisher_vol_spike_hma_1d1w_v1"
timeframe = "6h"
leverage = 1.0

def calculate_fisher_transform(high, low, close, period=9):
    """
    Ehlers Fisher Transform - transforms price into Gaussian-like distribution
    Excellent for catching reversals in range/bear markets
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    fisher = np.full(n, np.nan, dtype=np.float64)
    fisher_prev = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        # Find highest high and lowest low over period
        hh = np.max(high[i - period + 1:i + 1])
        ll = np.min(low[i - period + 1:i + 1])
        
        if hh == ll:
            continue
        
        # Normalize price to -1 to +1 range
        price_norm = 2.0 * (close[i] - ll) / (hh - ll) - 1.0
        
        # Clamp to avoid division by zero
        price_norm = np.clip(price_norm, -0.999, 0.999)
        
        # Fisher transform
        fisher[i] = 0.5 * np.log((1.0 + price_norm) / (1.0 - price_norm))
        
        if i > period - 1:
            fisher_prev[i] = fisher[i - 1]
    
    return fisher, fisher_prev

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_hma(close, period):
    """Hull Moving Average - reduces lag while smoothing"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = max(1, period // 2)
    sqrt_n = max(1, int(np.sqrt(period)))
    
    def wma(series, span):
        result = np.full(len(series), np.nan, dtype=np.float64)
        weights = np.arange(1, span + 1, dtype=np.float64)
        weight_sum = np.sum(weights)
        for i in range(span - 1, len(series)):
            if not np.isnan(series[i]):
                window = series[i - span + 1:i + 1].astype(np.float64)
                if not np.any(np.isnan(window)):
                    result[i] = np.sum(window * weights) / weight_sum
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff[i] = 2.0 * wma_half[i] - wma_full[i]
    
    return wma(diff, sqrt_n)

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF indicators
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 6h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    atr_7 = calculate_atr(high, low, close, period=7)
    atr_30 = calculate_atr(high, low, close, period=30)
    
    fisher, fisher_prev = calculate_fisher_transform(high, low, close, period=9)
    
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
    
    # Warmup period
    min_bars = 100
    
    for i in range(min_bars, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(fisher[i]) or np.isnan(fisher_prev[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === VOL SPIKE FILTER (ATR ratio) ===
        # Only enter when volatility is expanding (ATR7 > 1.8 * ATR30)
        vol_spike = False
        if not np.isnan(atr_7[i]) and not np.isnan(atr_30[i]) and atr_30[i] > 1e-10:
            vol_spike = (atr_7[i] / atr_30[i]) > 1.8
        
        # === TREND DIRECTION (Daily HMA) ===
        price_above_1d = close[i] > hma_1d_aligned[i]
        price_below_1d = close[i] < hma_1d_aligned[i]
        
        # Weekly HMA for additional confirmation
        hma_1w_valid = not np.isnan(hma_1w_aligned[i])
        price_above_1w = hma_1w_valid and close[i] > hma_1w_aligned[i]
        price_below_1w = hma_1w_valid and close[i] < hma_1w_aligned[i]
        
        # === FISHER TRANSFORM REVERSAL SIGNALS ===
        fisher_cross_up = fisher_prev[i] < -1.5 and fisher[i] >= -1.5
        fisher_cross_down = fisher_prev[i] > 1.5 and fisher[i] <= 1.5
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # LONG: Fisher crosses above -1.5 (oversold reversal) + uptrend + vol spike
        if fisher_cross_up and price_above_1d and vol_spike:
            if price_above_1w:
                desired_signal = SIZE_STRONG  # All timeframes aligned
            else:
                desired_signal = SIZE_BASE  # Basic setup
        
        # SHORT: Fisher crosses below +1.5 (overbought reversal) + downtrend + vol spike
        elif fisher_cross_down and price_below_1d and vol_spike:
            if price_below_1w:
                desired_signal = -SIZE_STRONG  # All timeframes aligned
            else:
                desired_signal = -SIZE_BASE  # Basic setup
        
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
        
        # === FISHER EXIT SIGNAL (opposite cross) ===
        fisher_exit_long = in_position and position_side > 0 and fisher[i] > 1.5
        fisher_exit_short = in_position and position_side < 0 and fisher[i] < -1.5
        
        if fisher_exit_long or fisher_exit_short:
            desired_signal = 0.0
        
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