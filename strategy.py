#!/usr/bin/env python3
"""
Experiment #003: 4h Donchian Breakout + 1d Regime + Extreme Volume + Low Choppiness

HYPOTHESIS: Donchian breakouts work when combined with strict regime filtering.
In bull regime (price > 1d SMA200): only long breakouts. In bear regime (price < 1d SMA200):
only short breakouts. This adapts to market conditions and avoids counter-trend trades
that destroy performance in 2022 crash and 2025 bear market.

WHY THIS SHOULD WORK IN BOTH BULL AND BEAR:
- Bull (2021, 2023-2024): Long breakouts with trend
- Bear (2022, 2025): Short breakouts with trend (no stubborn longs)
- Volume spike (2.5x) confirms institutional participation
- Choppiness < 40 filters out range-bound whipsaws
- One trade per Donchian cycle prevents re-entry churn

TARGET: 75-150 total trades over 4 years (19-37/year).
KEY DESIGN:
1. Donchian(20) breakout - price closes beyond 20-bar high/low
2. 1d SMA200 regime filter - direction must match regime
3. Volume spike > 2.5x 20-avg (very strict)
4. Choppiness < 40 (strong trend only)
5. ATR 2.5x stoploss
6. Signal: 0.30 (discrete)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_regime_vol_chop_1d_v2"
timeframe = "4h"
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

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures market choppiness
    CHOP > 61.8 = ranging, CHOP < 38.2 = trending
    We use < 40 for strong trend filter
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i - period + 1:i + 1])
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and atr_sum > 0:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop

def calculate_sma(close, period):
    """Simple Moving Average"""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def calculate_donchian_channels(high, low, period=20):
    """Donchian Channel - highest high and lowest low over period"""
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
    
    # Load 1d data for regime filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d SMA200 for regime
    sma_1d_raw = calculate_sma(df_1d['close'].values, 200)
    sma_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_1d_raw)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian_channels(high, low, period=20)
    
    # Volume moving average
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
    
    # Donchian cycle tracking (prevent re-entry until new cycle)
    last_donchian_break = 0  # 0=none, 1=upper, -1=lower
    
    # Warmup
    warmup = 220  # Need 200 for 1d SMA + 20 for Donchian
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(chop_14[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(vol_ratio[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(sma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME CHECK ===
        chop = chop_14[i]
        is_trending = chop < 40.0  # Very strict - only strong trends
        
        # === REGIME BIAS (1d SMA200) ===
        price_above_1d_sma = close[i] > sma_1d_aligned[i]
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 2.5  # Very strict - only extreme volume
        
        # === DONCHIAN BREAKOUT DETECTION ===
        # Breakout = close beyond Donchian channel
        upper_breakout = close[i] > donchian_upper[i - 1] if i > 0 else False
        lower_breakout = close[i] < donchian_lower[i - 1] if i > 0 else False
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # LONG: Upper breakout + bull regime + volume + trend
        if is_trending and price_above_1d_sma and upper_breakout and vol_spike:
            # Only enter if not already in a Donchian cycle
            if last_donchian_break != 1:
                desired_signal = SIZE
                last_donchian_break = 1
        
        # SHORT: Lower breakout + bear regime + volume + trend
        if is_trending and not price_above_1d_sma and lower_breakout and vol_spike:
            # Only enter if not already in a Donchian cycle
            if last_donchian_break != -1:
                desired_signal = -SIZE
                last_donchian_break = -1
        
        # Reset Donchian cycle when price returns to middle
        if in_position:
            mid_channel = (donchian_upper[i] + donchian_lower[i]) / 2
            if position_side > 0 and close[i] < mid_channel:
                last_donchian_break = 0
            elif position_side < 0 and close[i] > mid_channel:
                last_donchian_break = 0
        
        # === STOPLOSS CHECK ===
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
            last_donchian_break = 0
        
        # === REGIME EXIT ===
        # Exit if regime flips against position
        regime_exit = False
        if in_position and position_side > 0 and not price_above_1d_sma:
            regime_exit = True
        if in_position and position_side < 0 and price_above_1d_sma:
            regime_exit = True
        
        if regime_exit:
            desired_signal = 0.0
            last_donchian_break = 0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
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