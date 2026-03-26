#!/usr/bin/env python3
"""
Experiment #023: 4h Donchian Breakout + Volume + 1d Trend

HYPOTHESIS: Donchian(20) breakouts mark institutional accumulation/distribution zones.
Combined with volume confirmation (1.5x avg) and 1d HMA trend alignment, this captures
major trend moves. Entry ONLY on breakout bar (not continuation), preventing overtrading.
Exit via opposite Donchian band or 2.5 ATR stop. Works in both bull (long breakouts) and
bear (short breakdowns with bearish 1d trend).

TIMEFRAME: 4h primary
HTF: 1d for trend alignment
TARGET: 50-150 total trades over 4 years (12-37/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_vol_1d_trend_v3"
timeframe = "4h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average"""
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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d HMA for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate local 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Donchian 20-period
    period = 20
    donch_upper = np.full(n, np.nan, dtype=np.float64)
    donch_lower = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        donch_upper[i] = np.max(high[i - period + 1:i + 1])
        donch_lower[i] = np.min(low[i - period + 1:i + 1])
    
    # Volume MA for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = 50
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]):
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
        
        # === TREND BIAS (1d HMA) ===
        price_above_1d_hma = close[i] > hma_1d_aligned[i]
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === DONCHIAN BREAKOUT DETECTION (ONLY on breakout bar) ===
        # Breakout up: current close breaks ABOVE previous 20-bar high
        breakout_up = (close[i] > donch_upper[i-1]) if i > 1 else False
        # Breakout down: current close breaks BELOW previous 20-bar low
        breakout_down = (close[i] < donch_lower[i-1]) if i > 1 else False
        
        # === STOPLOSS CHECK (2.5 ATR) - processed first ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            in_position = False
            position_side = 0
            signals[i] = 0.0
            continue
        
        # === EXIT: Opposite Donchian band ===
        exit_triggered = False
        
        if in_position and position_side > 0:
            # Long exit: price breaks below lower band
            if close[i] < donch_lower[i-1] if i > 1 else False:
                exit_triggered = True
        
        if in_position and position_side < 0:
            # Short exit: price breaks above upper band
            if close[i] > donch_upper[i-1] if i > 1 else False:
                exit_triggered = True
        
        if exit_triggered:
            in_position = False
            position_side = 0
            signals[i] = 0.0
            continue
        
        # === ENTRY LOGIC ===
        if not in_position:
            # === NEW LONG ENTRY ===
            # Price breaks above previous 20-bar high + volume spike + bullish 1d trend
            if breakout_up and vol_spike and price_above_1d_hma:
                in_position = True
                position_side = 1
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                stop_price = entry_price - 2.5 * entry_atr
                signals[i] = SIZE
                continue
            
            # === NEW SHORT ENTRY ===
            # Price breaks below previous 20-bar low + volume spike + bearish 1d trend
            if breakout_down and vol_spike and not price_above_1d_hma:
                in_position = True
                position_side = -1
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                stop_price = entry_price + 2.5 * entry_atr
                signals[i] = -SIZE
                continue
        
        # === MAINTAIN POSITION ===
        if in_position:
            # Update highest/lowest for trailing stop
            if position_side > 0:
                highest_since_entry = max(highest_since_entry, high[i])
                trailing_stop = highest_since_entry - 2.5 * entry_atr
                stop_price = max(stop_price, trailing_stop)
            else:
                lowest_since_entry = min(lowest_since_entry, low[i])
                trailing_stop = lowest_since_entry + 2.5 * entry_atr
                stop_price = min(stop_price, trailing_stop)
            
            signals[i] = SIZE if position_side > 0 else -SIZE
    
    return signals