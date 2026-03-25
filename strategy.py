#!/usr/bin/env python3
"""
Experiment #1442: 4h Primary + 1d HTF — KAMA Adaptive Trend + ROC Momentum + Volume

Hypothesis: Based on #1431 success (6h KAMA+ROC+volume, Sharpe=0.381), this adapts
the winning formula to 4h timeframe with 1d trend filter. Key insights:
1. KAMA adapts to volatility better than HMA/EMA in crypto's choppy markets
2. ROC momentum confirms trend direction (worked in #1431)
3. Volume spike filter adds conviction (avoids false breakouts)
4. 1d KAMA filter prevents counter-trend trades in major moves
5. LOOSE entry conditions to guarantee 30+ trades (learned from 0-trade failures)

Why this should beat current best (Sharpe=0.575):
- 4h TF = more signals than 6h while maintaining fee efficiency
- Volume confirmation reduces whipsaw entries
- KAMA efficiency ratio adapts to regime changes automatically
- Simple logic = fewer conditions that can all fail simultaneously

Entry logic (LOOSE to guarantee trades):
- LONG: 1d_KAMA bullish + 4h_KAMA rising + ROC(10) > -5 + volume > 0.8*avg
- SHORT: 1d_KAMA bearish + 4h_KAMA falling + ROC(10) < 5 + volume > 0.8*avg

Target: Sharpe>0.6, trades>=40 train, trades>=5 test, DD>-30%
Timeframe: 4h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_roc_volume_1d_v1"
timeframe = "4h"
leverage = 1.0

def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average (KAMA)
    Adapts smoothing based on market efficiency (trend vs noise)
    From Perry Kaufman's "Trading Systems and Methods"
    """
    n = len(close)
    if n < period + slow_period:
        return np.full(n, np.nan)
    
    kama = np.full(n, np.nan, dtype=np.float64)
    
    # Change = absolute price change over period
    change = np.abs(close[period:] - close[:-period])
    
    # Sum of individual changes (volatility/noise)
    sum_changes = np.zeros(len(change))
    for i in range(len(change)):
        sum_changes[i] = np.sum(np.abs(np.diff(close[i:i+period+1])))
    
    # Efficiency Ratio (ER) = change / noise (0 to 1)
    er = np.zeros(len(change))
    mask = sum_changes != 0
    er[mask] = change[mask] / sum_changes[mask]
    er = np.clip(er, 0, 1)
    
    # Smoothing constants
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    # Adaptive smoothing constant
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Initialize KAMA
    kama[period - 1] = close[period - 1]
    
    # Calculate KAMA
    for i in range(period, n):
        idx = i - period
        if idx < len(sc):
            kama[i] = kama[i-1] + sc[idx] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    return kama

def calculate_roc(close, period=10):
    """Rate of Change - momentum indicator"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    roc = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        if close[i - period] != 0:
            roc[i] = 100.0 * (close[i] - close[i - period]) / close[i - period]
    
    return roc

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

def calculate_volume_sma(volume, period=20):
    """Simple moving average of volume"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_sma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return vol_sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF indicators
    kama_1d_raw = calculate_kama(df_1d['close'].values, period=10)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_raw)
    
    # Calculate 4h indicators
    kama_4h = calculate_kama(close, period=10)
    roc_10 = calculate_roc(close, period=10)
    atr_14 = calculate_atr(high, low, close, period=14)
    vol_sma_20 = calculate_volume_sma(volume, period=20)
    
    # KAMA slope (direction)
    kama_slope_4h = np.zeros(n)
    for i in range(1, n):
        if not np.isnan(kama_4h[i]) and not np.isnan(kama_4h[i-1]):
            kama_slope_4h[i] = kama_4h[i] - kama_4h[i-1]
    
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
        
        if np.isnan(kama_4h[i]) or np.isnan(kama_slope_4h[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(roc_10[i]) or np.isnan(vol_sma_20[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(kama_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === TREND DIRECTION (1d KAMA bias) ===
        price_above_1d = close[i] > kama_1d_aligned[i]
        price_below_1d = close[i] < kama_1d_aligned[i]
        
        # === 4h KAMA MOMENTUM (adaptive trend) ===
        kama_rising = kama_slope_4h[i] > 0
        kama_falling = kama_slope_4h[i] < 0
        
        # === ROC MOMENTUM (LOOSE threshold) ===
        roc = roc_10[i]
        roc_positive = roc > -5.0  # Very loose - allows slight pullbacks
        roc_negative = roc < 5.0   # Very loose - allows slight bounces
        
        # === VOLUME CONFIRMATION (LOOSE) ===
        vol_ratio = volume[i] / vol_sma_20[i] if vol_sma_20[i] > 0 else 0
        vol_confirmed = vol_ratio > 0.8  # Very loose - just not extremely low
        
        # === ENTRY LOGIC (LOOSE - must generate trades) ===
        desired_signal = 0.0
        
        # LONG: 1d bullish + 4h KAMA rising + ROC not too negative + volume OK
        if price_above_1d and kama_rising and roc_positive and vol_confirmed:
            # Strong if ROC also positive
            if roc > 0:
                desired_signal = SIZE_STRONG
            else:
                desired_signal = SIZE_BASE
        
        # SHORT: 1d bearish + 4h KAMA falling + ROC not too positive + volume OK
        elif price_below_1d and kama_falling and roc_negative and vol_confirmed:
            # Strong if ROC also negative
            if roc < 0:
                desired_signal = -SIZE_STRONG
            else:
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