#!/usr/bin/env python3
"""
Experiment #1520: 6h Primary + 1d/1w HTF — Volatility Cycle Strategy

Hypothesis: Volatility cycles work in BOTH bull and bear markets, unlike pure trend
following which died in 2022 crash and 2025 bear. This strategy captures the
"volatility crush" pattern: after vol expansion (panic/euphoria), vol contracts
and price mean-reverts. 

Key components:
1. 1w HMA(21) - major trend bias (only trade WITH weekly trend)
2. 1d ATR ratio (ATR7/ATR30) - detects vol expansion on daily
3. 6h ATR ratio (ATR7/ATR30) - entry trigger when vol contracts after expansion
4. 6h RSI(14) - entry timing (oversold for long, overbought for short)
5. 6h Bollinger Band width - confirms vol contraction
6. ATR(14) trailing stoploss (2.5x ATR)

Why this should work on 6h:
- 6h captures multi-day vol cycles (unlike 15m/1h noise)
- Weekly trend filter prevents counter-trend disasters
- Vol contraction after expansion = high-probability mean reversion
- LOOSE thresholds guarantee ≥30 trades/train, ≥3/test
- Works in bull (2021), bear (2022, 2025), and range (2023-2024)

Entry logic (LOOSE):
- LONG: 1w_HMA bullish + 1d_ATR_ratio > 1.3 + 6h_ATR_ratio < 1.2 + RSI < 50
- SHORT: 1w_HMA bearish + 1d_ATR_ratio > 1.3 + 6h_ATR_ratio < 1.2 + RSI > 50

Target: Sharpe>0.6, trades>=30 train, trades>=5 test, DD>-35%
Timeframe: 6h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_vol_cycle_hma_atr_rsi_1w1d_v1"
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

def calculate_atr_ratio(high, low, close, short_period=7, long_period=30):
    """ATR Ratio - detects vol expansion/contraction"""
    n = len(close)
    if n < long_period + 1:
        return np.full(n, np.nan)
    
    atr_short = calculate_atr(high, low, close, short_period)
    atr_long = calculate_atr(high, low, close, long_period)
    
    ratio = np.full(n, np.nan, dtype=np.float64)
    mask = (atr_long > 1e-10) & (~np.isnan(atr_short)) & (~np.isnan(atr_long))
    ratio[mask] = atr_short[mask] / atr_long[mask]
    
    return ratio

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

def calculate_bollinger_width(close, period=20):
    """Bollinger Band Width - (Upper - Lower) / Middle"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    width = np.full(n, np.nan, dtype=np.float64)
    mask = sma > 1e-10
    width[mask] = (2.0 * std[mask]) / sma[mask]
    
    return width

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
    
    atr_ratio_1d_raw = calculate_atr_ratio(
        df_1d['high'].values, 
        df_1d['low'].values, 
        df_1d['close'].values, 
        short_period=7, 
        long_period=30
    )
    atr_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio_1d_raw)
    
    # Calculate 6h indicators
    atr_ratio_6h = calculate_atr_ratio(high, low, close, short_period=7, long_period=30)
    rsi_14 = calculate_rsi(close, period=14)
    bb_width_20 = calculate_bollinger_width(close, period=20)
    atr_14 = calculate_atr(high, low, close, period=14)
    
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
        
        if np.isnan(rsi_14[i]) or np.isnan(atr_ratio_6h[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(bb_width_20[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1w_aligned[i]) or np.isnan(atr_ratio_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === VOLATILITY CYCLE DETECTION ===
        atr_ratio_1d = atr_ratio_1d_aligned[i]
        atr_ratio_6h_val = atr_ratio_6h[i]
        
        # Vol expansion on daily (panic/euphoria happened)
        vol_expanded_1d = atr_ratio_1d > 1.3
        
        # Vol contracting on 6h (vol crush pattern - entry signal)
        vol_contracting_6h = atr_ratio_6h_val < 1.2
        
        # Bollinger width confirms vol contraction
        bb_width_low = bb_width_20[i] < 0.08  # narrow bands
        
        # === TREND BIAS (1w HMA) ===
        price_above_1w = close[i] > hma_1w_aligned[i]
        price_below_1w = close[i] < hma_1w_aligned[i]
        
        # === RSI ENTRY TIMING ===
        rsi = rsi_14[i]
        rsi_oversold = rsi < 50
        rsi_overbought = rsi > 50
        
        # === ENTRY LOGIC (LOOSE - must generate trades) ===
        desired_signal = 0.0
        
        # LONG: Weekly bullish + vol expanded on 1d + vol contracting on 6h + RSI not overbought
        if price_above_1w and vol_expanded_1d and vol_contracting_6h and rsi_oversold:
            desired_signal = SIZE_STRONG
        
        # SHORT: Weekly bearish + vol expanded on 1d + vol contracting on 6h + RSI not oversold
        elif price_below_1w and vol_expanded_1d and vol_contracting_6h and rsi_overbought:
            desired_signal = -SIZE_STRONG
        
        # Weaker signals when BB width confirms (add confluence)
        elif price_above_1w and vol_contracting_6h and bb_width_low and rsi < 45:
            desired_signal = SIZE_BASE
        
        elif price_below_1w and vol_contracting_6h and bb_width_low and rsi > 55:
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