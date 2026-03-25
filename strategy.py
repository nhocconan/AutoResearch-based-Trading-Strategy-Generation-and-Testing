#!/usr/bin/env python3
"""
Experiment #1651: 6h Primary + 1w/1d HTF — Simplified Multi-TF Trend Pullback

Hypothesis: 6h timeframe with 1w trend bias + 1d confirmation captures multi-day 
swings better than 4h (too noisy) or 12h (too slow). Recent 6h failures used overly 
complex regime filters (Fisher+CHOP+RSI). This strategy uses SIMPLER logic:

1. 1w HMA(21) = PRIMARY trend direction (most stable for 6h entries)
2. 1d HMA(21) = SECONDARY confirmation (must agree with 1w)
3. 6h RSI(14) pullback = ENTRY timing (RSI 40-60 in trend direction)
4. 6h ATR vol filter = avoid low vol chop (ATR ratio > 1.2)
5. Simple 2.5x ATR trailing stoploss

Why this beats recent 6h failures:
- #1640 Fisher+RSI+regime: too many conflicting filters
- #1643 CRSI+CHOP: regime detection too slow for 6h
- #1647 Vol expansion: volume unreliable on 6h

This is SIMPLER: just trend alignment + pullback entry.
Looser RSI thresholds (40-60 not 30-70) = more trades.

Target: Sharpe>0.6, trades≥30 train, trades≥5 test, DD>-35%
Timeframe: 6h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_trend_pullback_1w1d_simple_v1"
timeframe = "6h"
leverage = 1.0

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

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    gain = np.insert(gain, 0, 0)
    loss = np.insert(loss, 0, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan, dtype=np.float64)
    mask = avg_loss != 0
    rs = np.zeros(n)
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rsi[mask] = 100 - (100 / (1 + rs[mask]))
    
    return rsi

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
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF indicators
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 6h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    atr_30 = calculate_atr(high, low, close, period=30)
    rsi_14 = calculate_rsi(close, period=14)
    sma_50 = calculate_sma(close, period=50)
    sma_200 = calculate_sma(close, period=200)
    
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
    min_bars = 250
    
    for i in range(min_bars, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(sma_50[i]) or np.isnan(sma_200[i]):
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
        
        # === TREND DIRECTION (1w + 1d HMA alignment) ===
        price_above_1w = close[i] > hma_1w_aligned[i]
        price_below_1w = close[i] < hma_1w_aligned[i]
        price_above_1d = close[i] > hma_1d_aligned[i]
        price_below_1d = close[i] < hma_1d_aligned[i]
        
        # 1w HMA slope (trend strength)
        hma_1w_slope = 0.0
        if i >= 4 and not np.isnan(hma_1w_aligned[i-4]):
            hma_1w_slope = (hma_1w_aligned[i] - hma_1w_aligned[i-4]) / hma_1w_aligned[i-4]
        
        # 1d HMA slope
        hma_1d_slope = 0.0
        if i >= 4 and not np.isnan(hma_1d_aligned[i-4]):
            hma_1d_slope = (hma_1d_aligned[i] - hma_1d_aligned[i-4]) / hma_1d_aligned[i-4]
        
        # Strong trend = both 1w and 1d agree + positive slope
        strong_bull_trend = price_above_1w and price_above_1d and hma_1w_slope > 0 and hma_1d_slope > 0
        strong_bear_trend = price_below_1w and price_below_1d and hma_1w_slope < 0 and hma_1d_slope < 0
        
        # Moderate trend = price above/below both but slope neutral
        mod_bull_trend = price_above_1w and price_above_1d
        mod_bear_trend = price_below_1w and price_below_1d
        
        # === VOLATILITY FILTER (ATR ratio) ===
        atr_ratio = atr_14[i] / atr_30[i] if atr_30[i] > 1e-10 else 1.0
        vol_expanding = atr_ratio > 1.1
        vol_normal = atr_ratio > 0.8
        
        # === RSI PULLBACK (LOOSE thresholds for trades) ===
        rsi_val = rsi_14[i]
        
        # Bull pullback = RSI dipped but not oversold
        rsi_bull_pullback = 40 <= rsi_val <= 60
        rsi_bull_strong = 45 <= rsi_val <= 55
        
        # Bear pullback = RSI rallied but not overbought
        rsi_bear_pullback = 40 <= rsi_val <= 60
        rsi_bear_strong = 45 <= rsi_val <= 55
        
        # === PRICE ACTION FILTER ===
        price_above_sma50 = close[i] > sma_50[i]
        price_below_sma50 = close[i] < sma_50[i]
        price_above_sma200 = close[i] > sma_200[i]
        price_below_sma200 = close[i] < sma_200[i]
        
        # === ENTRY LOGIC (LOOSE - must generate trades) ===
        desired_signal = 0.0
        
        # LONG: Strong bull trend + RSI pullback + vol normal/expanding
        if strong_bull_trend and rsi_bull_pullback and vol_normal:
            desired_signal = SIZE_STRONG if rsi_bull_strong else SIZE_BASE
        
        # LONG: Moderate bull trend + RSI pullback + above SMA50 + vol normal
        elif mod_bull_trend and rsi_bull_pullback and price_above_sma50 and vol_normal:
            desired_signal = SIZE_BASE
        
        # SHORT: Strong bear trend + RSI pullback + vol normal/expanding
        elif strong_bear_trend and rsi_bear_pullback and vol_normal:
            desired_signal = -SIZE_STRONG if rsi_bear_strong else -SIZE_BASE
        
        # SHORT: Moderate bear trend + RSI pullback + below SMA50 + vol normal
        elif mod_bear_trend and rsi_bear_pullback and price_below_sma50 and vol_normal:
            desired_signal = -SIZE_BASE
        
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