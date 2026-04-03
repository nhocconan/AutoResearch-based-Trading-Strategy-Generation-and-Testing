#!/usr/bin/env python3
"""
Experiment #018: 1d Donchian(20) Breakout + 1w HMA Trend + Volume Spike

HYPOTHESIS: Combining 1d Donchian breakouts with 1w HMA trend alignment and volume confirmation 
creates a robust signal for capturing medium-term trends while minimizing whipsaw. The strategy 
enters long when price breaks above the 20-period Donchian channel on 1d with bullish 1w trend 
and volume spike, and shorts on breakdown below the Donchian channel with bearish 1w trend 
and volume confirmation. Uses ATR-based stoploss for risk control. Designed for low trade 
frequency (target: 30-100 trades over 4 years) to reduce fee drag and improve generalization 
across bull/bear markets.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_donchian_1w_hma_volume_v1"
timeframe = "1d"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average calculation."""
    if len(close) < period:
        return np.full_like(close, np.nan, dtype=np.float64)
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    # WMA helper
    def wma(values, window):
        if len(values) < window:
            return np.full_like(values, np.nan, dtype=np.float64)
        weights = np.arange(1, window + 1, dtype=np.float64)
        return np.convolve(values, weights[::-1], mode='valid') / weights.sum()
    
    # Calculate WMAs
    wma_half = wma(close, half_period)
    wma_full = wma(close, period)
    
    # Ensure we have enough data for calculations
    if len(wma_half) < half_period or len(wma_full) < 1:
        return np.full_like(close, np.nan, dtype=np.float64)
    
    # Align arrays (WMA_half starts at index half_period-1, WMA_full at period-1)
    raw_hma = 2 * wma_half[-len(wma_full):] - wma_full
    
    # Final WMA of raw_hma
    hma = wma(raw_hma, sqrt_period)
    
    # Pad with NaN to match original length
    result = np.full_like(close, np.nan, dtype=np.float64)
    start_idx = period - 1  # WMA_full starts here
    hma_start = half_period + sqrt_period - 1  # Compensate for all WMA delays
    if hma_start < len(result) and len(hma) > 0:
        end_idx = min(start_idx + len(hma), len(result))
        result[hma_start:end_idx] = hma[:end_idx - hma_start]
    
    return result

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w OHLC for HMA trend (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    hma_21 = calculate_hma(df_1w['close'].values, 21)
    hma_21_aligned = align_htf_to_ltf(prices, df_1w, hma_21)  # shift(1) for completed bars
    
    # === 1d Indicators ===
    atr_14 = pd.Series(high).rolling(window=14, min_periods=14).apply(
        lambda x: max(x[:,0] - x[:,1], abs(x[:,0] - np.roll(x[:,1], 1)), abs(x[:,1] - np.roll(x[:,0], 1))).mean(),
        raw=False
    ).values if len(high) >= 14 else np.full(n, np.nan)
    
    # Simplified ATR calculation
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # Donchian Channel (20-period)
    dc_upper_20 = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    dc_lower_20 = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Volume confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_bar = -1
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    warmup = 50  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(atr_14[i]) or np.isnan(dc_upper_20[i]) or np.isnan(dc_lower_20[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(hma_21_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Price Levels ---
        hma_21_val = hma_21_aligned[i]
        
        # --- 1w Trend Filter ---
        trend_bullish = close[i] > hma_21_val
        trend_bearish = close[i] < hma_21_val
        
        # --- Price Channel Breakout ---
        bullish_breakout = close[i] > dc_upper_20[i]
        bearish_breakout = close[i] < dc_lower_20[i]
        
        # --- Volume Confirmation ---
        vol_ok = volume[i] > vol_ma_20[i] * 1.5 if vol_ma_20[i] > 1e-10 else False  # 1.5x volume spike
        
        # --- Position Management (Exit Logic) ---
        stop_hit = False
        
        if in_position:
            # ATR-based trailing stoploss
            if position_side > 0:
                stop_level = highest_since_entry - 2.5 * atr_14[i]
                if low[i] < stop_level:
                    stop_hit = True
            else:  # Short position
                stop_level = lowest_since_entry + 2.5 * atr_14[i]
                if high[i] > stop_level:
                    stop_hit = True
            
            # Exit conditions: trend reversal or opposite Donchian touch
            min_hold = (i - entry_bar) >= 2  # Minimum 2 bars hold
            if min_hold:
                if position_side > 0:
                    # Exit long: trend turns bearish OR price touches lower Donchian
                    if trend_bearish or close[i] <= dc_lower_20[i]:
                        stop_hit = True
                else:  # position_side < 0
                    # Exit short: trend turns bullish OR price touches upper Donchian
                    if trend_bullish or close[i] >= dc_upper_20[i]:
                        stop_hit = True
            
            if stop_hit:
                signals[i] = 0.0
                in_position = False
                position_side = 0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
            else:
                signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long conditions: 
        # Breakout above upper Donchian with bullish 1w trend AND volume confirmation
        if bullish_breakout and trend_bullish and vol_ok:
            in_position = True
            position_side = 1
            entry_bar = i
            highest_since_entry = high[i]
            signals[i] = SIZE
        # Short conditions:
        # Breakout below lower Donchian with bearish 1w trend AND volume confirmation
        elif bearish_breakout and trend_bearish and vol_ok:
            in_position = True
            position_side = -1
            entry_bar = i
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals