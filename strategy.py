#!/usr/bin/env python3
"""
Experiment #023: 12h Donchian Breakout + 1d/1w Trend Confirmation

HYPOTHESIS: 12h Donchian(20) breakouts mark institutional moves. Using:
- 1d HMA for immediate trend alignment
- 1w HMA for regime context
- Volume spike confirmation (1.5x)
- ATR trailing stop (3x)

This should work in BOTH bull (long breakouts) and bear (short breakdowns + short rallies).
12h is slow enough to avoid overtrading while capturing major trend moves.
Target: 75-150 total trades over 4 years (19-37/year).
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_vol_1d_1w_v1"
timeframe = "12h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = max(1, period // 2)
    sqrt_n = max(1, int(np.sqrt(period)))
    
    def wma(series, span):
        result = np.full(len(series), np.nan)
        weights = np.arange(1, span + 1)
        weight_sum = span * (span + 1) // 2
        
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1]
            if not np.any(np.isnan(window)):
                result[i] = np.sum(window * weights) / weight_sum
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.where(np.isnan(wma_half) | np.isnan(wma_full), np.nan, 2.0 * wma_half - wma_full)
    
    return wma(diff, sqrt_n)

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # EMA-based ATR
    atr = np.full(n, np.nan)
    sma = np.mean(tr[:period])
    multiplier = 2.0 / (period + 1)
    
    for i in range(period - 1, n):
        if i == period - 1:
            atr[i] = sma
        elif i > period - 1:
            atr[i] = atr[i-1] + multiplier * (tr[i] - atr[i-1])
    
    return atr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE ===
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # 1d HMA for trend
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # 1w HMA for regime
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Local 12h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Donchian 20
    donch_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume MA
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    trailing_high = 0.0
    trailing_low = 0.0
    bars_held = 0
    stop_price = 0.0
    
    warmup = 50
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 0:
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
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === CONDITION COMPONENTS ===
        # Volume spike
        vol_spike = vol_ratio[i] > 1.5
        
        # 1d trend: price above HMA21
        price_above_1d = close[i] > hma_1d_aligned[i]
        
        # 1w regime: price above HMA21
        price_above_1w = close[i] > hma_1w_aligned[i]
        
        # Donchian breakout detection
        breakout_up = close[i] > donch_upper[i]
        breakout_down = close[i] < donch_lower[i]
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG: Breakout above + volume + bullish 1d trend ===
            # In bull regime (price > 1w HMA): require breakout + vol + trend
            # In bear regime: skip longs (too risky)
            if breakout_up and vol_spike and price_above_1d and price_above_1w:
                desired_signal = SIZE
            
            # === SHORT: Breakout below + volume + bearish 1d trend ===
            # In bear regime: require breakout + vol + trend
            # In bull regime: skip shorts (counter-trend)
            if breakout_down and vol_spike and not price_above_1d and not price_above_1w:
                desired_signal = -SIZE
        
        # === EXIT LOGIC ===
        exit_triggered = False
        
        if in_position and position_side > 0:
            # Long: trailing stop OR breakout below lower band
            trailing_high = max(trailing_high, high[i])
            stop_price = trailing_high - 3.0 * entry_atr
            
            if low[i] < stop_price:
                exit_triggered = True
            elif breakout_down:
                exit_triggered = True
        
        if in_position and position_side < 0:
            # Short: trailing stop OR breakout above upper band
            trailing_low = min(trailing_low, low[i])
            stop_price = trailing_low + 3.0 * entry_atr
            
            if high[i] > stop_price:
                exit_triggered = True
            elif breakout_up:
                exit_triggered = True
        
        if exit_triggered:
            desired_signal = 0.0
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New entry or flip
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                trailing_high = high[i]
                trailing_low = low[i]
                bars_held = 0
                
                if position_side > 0:
                    stop_price = entry_price - 3.0 * entry_atr
                else:
                    stop_price = entry_price + 3.0 * entry_atr
            else:
                # Same direction - maintain
                bars_held += 1
        else:
            if in_position:
                in_position = False
                position_side = 0
                bars_held = 0
                trailing_high = 0.0
                trailing_low = 0.0
                stop_price = 0.0
        
        signals[i] = desired_signal
    
    return signals