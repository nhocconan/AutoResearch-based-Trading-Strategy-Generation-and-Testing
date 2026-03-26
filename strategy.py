#!/usr/bin/env python3
"""
Experiment #024: 4h Donchian Breakout + Volume Spike + 1d HMA Trend Filter

HYPOTHESIS: Donchian(20) breakouts mark institutional support/resistance breaks.
Volume spike ≥ 1.5x confirms institutional participation. 1d HMA trend filter 
prevents countertrend entries. Works in bull (long breakouts above 1d HMA) and 
bear (short breakouts below 1d HMA). ATR-based stoploss for risk control.

TIMEFRAME: 4h primary
HTF: 1d for trend filter (HMA alignment)
TARGET: 75-150 total trades per symbol over 4 years
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_vol_1d_hma_v2"
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
        tr[i] = max(high[i] - low[i], 
                    abs(high[i] - close[i-1]), 
                    abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_donchian(high, low, period=20):
    """Donchian Channel - returns upper and lower bands"""
    n = len(high)
    upper = np.full(n, np.nan, dtype=np.float64)
    lower = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d HMA for trend filter
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # === Calculate local 4h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Donchian 20-period
    donch_upper, donch_lower = calculate_donchian(high, low, period=20)
    
    # Previous period Donchian (shift by 1 to get the CLOSED bar's values)
    donch_upper_prev = pd.Series(donch_upper).shift(1).values
    donch_lower_prev = pd.Series(donch_lower).shift(1).values
    
    # Volume MA and ratio
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_ratio = volume / np.where(vol_ma.values > 0, vol_ma.values, 1)
    
    signals = np.zeros(n)
    SIZE = 0.30  # Position size (30% of capital)
    
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
        
        if np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]) or np.isnan(donch_upper_prev[i]):
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
        
        # === 1d HMA TREND FILTER ===
        price_above_1d_hma = close[i] > hma_1d_aligned[i]
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] >= 1.5
        
        # === DONCHIAN BREAKOUT DETECTION ===
        # Breakout: close CLOSES above previous upper band (not just touching)
        breakout_up = (close[i] > donch_upper_prev[i]) and (close[i-1] <= donch_upper_prev[i-1] if i > warmup else True)
        # Breakdown: close CLOSES below previous lower band
        breakout_down = (close[i] < donch_lower_prev[i]) and (close[i-1] >= donch_lower_prev[i-1] if i > warmup else True)
        
        # Current channel position
        price_above_upper = close[i] > donch_upper[i]
        price_below_lower = close[i] < donch_lower[i]
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === NEW LONG ENTRY ===
            # Breakout above previous channel + volume spike + bullish 1d trend
            if breakout_up or price_above_upper:
                if vol_spike and price_above_1d_hma:
                    desired_signal = SIZE
            
            # === NEW SHORT ENTRY ===
            # Breakdown below previous channel + volume spike + bearish 1d trend
            if breakout_down or price_below_lower:
                if vol_spike and not price_above_1d_hma:
                    desired_signal = -SIZE
        
        # === STOPLOSS CHECK (2.5 ATR trailing stop) ===
        if in_position and not stoploss_triggered:
            if position_side > 0:
                highest_since_entry = max(highest_since_entry, high[i])
                trailing_stop = highest_since_entry - 2.5 * entry_atr
                stop_price = max(stop_price, trailing_stop)
                if low[i] < stop_price:
                    stoploss_triggered = True
            
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, low[i])
                trailing_stop = lowest_since_entry + 2.5 * entry_atr
                stop_price = min(stop_price, trailing_stop)
                if high[i] > stop_price:
                    stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === STRUCTURAL EXIT (opposite channel break) ===
        exit_triggered = False
        
        if in_position and position_side > 0:
            # Long exit: price breaks below lower channel OR strong down move
            if price_below_lower:
                exit_triggered = True
        
        if in_position and position_side < 0:
            # Short exit: price breaks above upper channel OR strong up move
            if price_above_upper:
                exit_triggered = True
        
        if exit_triggered:
            desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New position or flip
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
                stoploss_triggered = False
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
                stoploss_triggered = False
        
        signals[i] = desired_signal
    
    return signals