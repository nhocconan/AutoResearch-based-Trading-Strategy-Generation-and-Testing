#!/usr/bin/env python3
"""
Experiment #010: 1h Primary + 4h HTF — Simplified MTF Trend Following

Hypothesis: Previous 1h/30m strategies (#005, #008) failed with 0 trades due to OVER-FILTERING.
This strategy SIMPLIFIES significantly to ensure trade generation while keeping MTF edge:

1. 4h HMA(21) for TREND DIRECTION only (not hard filter - just bias weighting)
2. 1h RSI(14) for ENTRY TIMING with LOOSE thresholds (30/70 not 20/80)
3. Volume filter LOOSENED (0.5x avg not 0.8x)
4. NO session filter (killed trades in #005, #008)
5. NO funding data dependency (removes complexity, ensures trades)
6. ATR(14) trailing stop at 2.5x for risk management
7. Discrete signal levels: 0.0, ±0.25, ±0.30

Key changes from failed 1h attempts:
- REMOVED session filter (was killing 60% of potential trades)
- REMOVED funding data (complexity without clear edge)
- LOOSENED RSI thresholds (30/70 vs 20/80)
- LOOSENED volume filter (0.5x vs 0.8x)
- 4h HMA for BIAS not BINARY filter (trade both directions with size weighting)

Entry Logic:
- 4h HMA bullish: prefer longs on 1h RSI<35, allow shorts only on RSI>75
- 4h HMA bearish: prefer shorts on 1h RSI>65, allow longs only on RSI<25
- Size: 0.30 with HTF trend, 0.20 against HTF trend

Risk: 2.5x ATR trailing stop, max signal magnitude 0.35
Target: Sharpe > 0.15, trades > 30/symbol train, > 3/symbol test, DD > -40%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_rsi_4h_hma_simplified_v1"
timeframe = "1h"
leverage = 1.0

def calculate_rsi(close, period=14):
    """
    Relative Strength Index (RSI)
    RSI = 100 - (100 / (1 + RS))
    RS = average gain / average loss over period
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    rsi = np.full(n, np.nan)
    delta = np.diff(close)
    
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    # Pad to match length
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    for i in range(period, n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

def calculate_hma(close, period=21):
    """
    Hull Moving Average (HMA)
    HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    Faster response than EMA with less lag
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    # WMA helper
    def wma(series, span):
        result = np.full(len(series), np.nan)
        weights = np.arange(1, span + 1)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1]
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half_period)
    wma_full = wma(close, period)
    
    # Combine
    raw_hma = 2.0 * wma_half - wma_full
    hma = wma(raw_hma, sqrt_period)
    
    return hma

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

def calculate_sma(close, period=20):
    """Simple Moving Average for volume smoothing"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    sma = np.full(n, np.nan)
    cumsum = np.cumsum(close)
    sma[period-1:] = (cumsum[period-1:] - np.concatenate([[0], cumsum[:-period]])) / period
    
    return sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate and align 4h HMA for trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate primary (1h) indicators
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    vol_sma = calculate_sma(volume, period=20)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25
    TREND_SIZE = 0.30
    MAX_SIZE = 0.35
    
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
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(rsi[i]) or np.isnan(vol_sma[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF TREND BIAS (4h HMA) ===
        hma_4h_bull = close[i] > hma_4h_aligned[i]
        hma_4h_bear = close[i] < hma_4h_aligned[i]
        
        # === VOLUME FILTER (LOOSENED) ===
        vol_ok = volume[i] > 0.5 * vol_sma[i] if vol_sma[i] > 0 else True
        
        # === DESIRED SIGNAL BASED ON RSI + HTF BIAS ===
        desired_signal = 0.0
        signal_strength = 0.0
        
        # LONG entries
        if rsi[i] < 35.0 and vol_ok:
            if hma_4h_bull:
                # With trend: larger size
                signal_strength = TREND_SIZE
            else:
                # Against trend: smaller size
                signal_strength = BASE_SIZE
            desired_signal = signal_strength
        
        # SHORT entries
        elif rsi[i] > 65.0 and vol_ok:
            if hma_4h_bear:
                # With trend: larger size
                signal_strength = TREND_SIZE
            else:
                # Against trend: smaller size
                signal_strength = BASE_SIZE
            desired_signal = -signal_strength
        
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
        # Clamp to max magnitude and discretize
        desired_signal = np.clip(desired_signal, -MAX_SIZE, MAX_SIZE)
        
        if desired_signal >= TREND_SIZE * 0.85:
            final_signal = TREND_SIZE
        elif desired_signal <= -TREND_SIZE * 0.85:
            final_signal = -TREND_SIZE
        elif desired_signal >= BASE_SIZE * 0.85:
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