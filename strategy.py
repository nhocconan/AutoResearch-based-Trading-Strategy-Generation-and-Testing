#!/usr/bin/env python3
"""
Experiment #013: 12h ATR Volatility Expansion + Volume + 1d HMA Trend

HYPOTHESIS: Major price moves are preceded by periods of low volatility (the "calm
before the storm"). When ATR contracts below its 50-period MA and then expands
above it with volume confirmation, institutional capital is entering. Combined
with 1d HMA for trend direction, this captures explosive moves in both bull
(volatility expansion with price above HMA = strong bullish continuation) and
bear (volatility expansion with price below HMA = capitulation/continuation).

This is DIFFERENT from my failed attempts (#009, #019, #020) which:
- #009: Used ATR ratio but with 0 trades (conditions too strict)
- #019: Used Camarilla + chop which underperformed  
- #020: Used BB squeeze which overtraded (218 trades)

KEY INSIGHT: Volatility expansion is a PROVEN precursor to big moves. Unlike
Donchian breakouts which can be whipsawed, ATR expansion with volume filters
only triggers when volatility itself is increasing.

TIMEFRAME: 12h primary
HTF: 1d for trend direction
TARGET: 75-150 total trades over 4 years (19-37/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_volatility_expansion_1d_v1"
timeframe = "12h"
leverage = 1.0

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
    
    # 1d HMA for trend direction
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # === Calculate 12h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # ATR-based volatility regime (key indicator)
    # When ATR is expanding relative to its MA, volatility is increasing
    atr_ma = pd.Series(atr_14).rolling(window=50, min_periods=50).mean().values
    atr_ratio = atr_14 / (atr_ma + 1e-10)
    
    # Donchian channel for structure
    donch_upper, donch_lower = calculate_donchian(high, low, period=20)
    donch_mid = (donch_upper + donch_lower) / 2.0
    
    # Volume confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / (vol_ma + 1e-10)
    
    # === Signal variables ===
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
    vol_expansion_bars = 0  # Count consecutive bars of ATR expansion
    
    warmup = 100  # Need enough bars for ATR(14), ATR MA(50), Donchian(20)
    
    for i in range(warmup, n):
        # Skip if key indicators not ready
        if np.isnan(atr_14[i]) or np.isnan(atr_ma[i]) or np.isnan(hma_1d_aligned[i]):
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
        
        # Current values
        current_atr_ratio = atr_ratio[i]
        current_vol_ratio = vol_ratio[i]
        price = close[i]
        trend_bullish = price > hma_1d_aligned[i]
        trend_bearish = price < hma_1d_aligned[i]
        
        # === VOLATILITY EXPANSION DETECTION ===
        # ATR expanding means volatility is increasing (good for momentum trades)
        # Key threshold: ATR ratio > 1.0 means current ATR is above its 50-bar MA
        atr_expanding = current_atr_ratio > 1.0
        
        # Count consecutive bars of expansion (need at least 1 bar)
        if atr_expanding:
            vol_expansion_bars += 1
        else:
            vol_expansion_bars = 0
        
        # === PRICE CHANNEL POSITION ===
        # Price relative to Donchian channel
        channel_width = donch_upper[i] - donch_lower[i]
        price_in_upper_half = price > donch_mid[i]
        
        # === ENTRY CONDITIONS ===
        desired_signal = 0.0
        
        if not in_position:
            # === NEW LONG ENTRY ===
            # Conditions:
            # 1. Volatility expanding (ATR ratio > 1.0)
            # 2. Volume confirming (1.3x+)
            # 3. Price above 1d HMA (bullish trend)
            # 4. Price in upper half of Donchian (not too extended)
            if atr_expanding and current_vol_ratio > 1.3:
                if trend_bullish and price_in_upper_half:
                    desired_signal = SIZE
        
        if not in_position:
            # === NEW SHORT ENTRY ===
            # Conditions:
            # 1. Volatility expanding (ATR ratio > 1.0)
            # 2. Volume confirming (1.3x+)
            # 3. Price below 1d HMA (bearish trend)
            # 4. Price in lower half of Donchian (not too extended)
            if atr_expanding and current_vol_ratio > 1.3:
                if trend_bearish and not price_in_upper_half:
                    desired_signal = -SIZE
        
        # === STOPLOSS CHECK (3 ATR) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 3.0 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 3.0 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
            vol_expansion_bars = 0  # Reset on stoploss
        
        # === EXIT: Trend reversal ===
        exit_triggered = False
        
        if in_position and position_side > 0:
            # Long exit: price crosses below 1d HMA (trend reversal)
            if close[i] < hma_1d_aligned[i]:
                exit_triggered = True
        
        if in_position and position_side < 0:
            # Short exit: price crosses above 1d HMA (trend reversal)
            if close[i] > hma_1d_aligned[i]:
                exit_triggered = True
        
        if exit_triggered:
            desired_signal = 0.0
            vol_expansion_bars = 0  # Reset on exit
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New position or direction flip
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                if position_side > 0:
                    stop_price = entry_price - 3.0 * entry_atr
                else:
                    stop_price = entry_price + 3.0 * entry_atr
            else:
                # Same direction - maintain position
                pass
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = desired_signal
    
    return signals