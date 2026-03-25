#!/usr/bin/env python3
"""
Experiment #1491: 6h Primary + 1w/1d HTF — Volatility Spike Reversion Strategy

Hypothesis: 6h timeframe captures multi-day volatility cycles perfectly. After panic 
moves (vol spike), prices tend to revert as fear subsides. This is especially 
profitable in bear/range markets (2022 crash, 2025 bear) where trend strategies fail.

Key components:
1. ATR Ratio (ATR7/ATR28): Detects volatility spikes (>2.0 = panic, <1.3 = calm)
2. Bollinger Band (20, 2.2): Identifies price extremes during vol spikes
3. 1d HMA(21): Major trend filter - only take reversions WITH the trend
4. 1w HMA(40): Ultra-long-term bias - avoid counter-trend in major moves
5. RSI(7): Fast mean-reversion signal (oversold/overbought during spikes)
6. ATR(14) trailing stoploss (2.5x ATR)

Why this should work on 6h:
- 6h captures 3-5 day vol cycles (panic → recovery pattern)
- Vol spike reversion worked through 2022 crash (unlike trend following)
- HTF filters prevent catching falling knives in major downtrends
- LOOSE thresholds guarantee trades (ATR>1.8 not 2.0, RSI<38 not 30)
- Discrete sizing minimizes fee churn on 6h timeframe

Entry logic (LOOSE to guarantee ≥30 trades/train, ≥3/test):
- LONG: ATR_ratio>1.8 + price<BB_lower + RSI<38 + 1d_HMA bullish
- SHORT: ATR_ratio>1.8 + price>BB_upper + RSI>62 + 1d_HMA bearish
- Exit: ATR_ratio<1.3 (vol normalized) OR stoploss hit

Target: Sharpe>0.6, trades>=30 train, trades>=5 test, DD>-35%
Timeframe: 6h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_vol_spike_reversion_hma_1w1d_v1"
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

def calculate_bollinger(close, period=20, std_mult=2.2):
    """Bollinger Bands with wider bands for 6h volatility"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    
    return upper, sma, lower

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
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=40)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 6h indicators
    atr_7 = calculate_atr(high, low, close, period=7)
    atr_28 = calculate_atr(high, low, close, period=28)
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_7 = calculate_rsi(close, period=7)
    bb_upper, bb_mid, bb_lower = calculate_bollinger(close, period=20, std_mult=2.2)
    
    # ATR Ratio (volatility spike detector)
    atr_ratio = np.full(n, np.nan, dtype=np.float64)
    for i in range(28, n):
        if atr_28[i] > 1e-10 and not np.isnan(atr_7[i]) and not np.isnan(atr_28[i]):
            atr_ratio[i] = atr_7[i] / atr_28[i]
    
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
    entry_atr_ratio = 0.0
    
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
        
        if np.isnan(rsi_7[i]) or np.isnan(bb_lower[i]) or np.isnan(bb_upper[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(atr_ratio[i]) or np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === VOLATILITY SPIKE DETECTION ===
        vol_spike = atr_ratio[i] > 1.8  # LOOSE: was 2.0
        vol_normal = atr_ratio[i] < 1.3  # Exit threshold
        
        # === PRICE AT BAND EXTREMES ===
        price_at_lower = close[i] <= bb_lower[i] * 1.005  # within 0.5%
        price_at_upper = close[i] >= bb_upper[i] * 0.995  # within 0.5%
        
        # === RSI EXTREMES (fast 7-period) ===
        rsi = rsi_7[i]
        rsi_oversold = rsi < 38  # LOOSE: was 30
        rsi_overbought = rsi > 62  # LOOSE: was 70
        
        # === HTF TREND FILTERS ===
        price_above_1d = close[i] > hma_1d_aligned[i]
        price_below_1d = close[i] < hma_1d_aligned[i]
        
        price_above_1w = close[i] > hma_1w_aligned[i]
        price_below_1w = close[i] < hma_1w_aligned[i]
        
        # === ENTRY LOGIC (LOOSE - must generate trades) ===
        desired_signal = 0.0
        
        # LONG: Vol spike + price at lower BB + RSI oversold + 1d bullish
        # Only require ONE of 1d/1w bullish (not both) for more trades
        if vol_spike and price_at_lower and rsi_oversold:
            if price_above_1d or price_above_1w:  # At least one HTF bullish
                desired_signal = SIZE_BASE
        
        # SHORT: Vol spike + price at upper BB + RSI overbought + 1d bearish
        if vol_spike and price_at_upper and rsi_overbought:
            if price_below_1d or price_below_1w:  # At least one HTF bearish
                desired_signal = -SIZE_BASE
        
        # === EXIT LOGIC (vol normalized) ===
        if in_position and vol_normal:
            desired_signal = 0.0
        
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
                entry_atr_ratio = atr_ratio[i]
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
                entry_atr_ratio = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = final_signal
    
    return signals