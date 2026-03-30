#!/usr/bin/env python3
"""
Experiment #028: 12h ATR-Normalized Donchian + KAMA Trend + Volume

HYPOTHESIS: Standard Donchian uses fixed periods, missing volatility context.
Using ATR-normalized channels adapts to market conditions - wider channels in
volatile 2022 crash, tighter in calm 2021 bull. Combined with 1d KAMA for trend
direction and volume confirmation, this should generate fewer but higher-quality
signals.

KEY INSIGHT: ATR(14) on 12h ≈ 7-day channel. This matches the 1d KAMA(10) well.
Both use ~10-period windows, so they're measuring the same trend cycle.

WHY 12h: Slower than 4h = fewer but higher-quality trades. 12-37 trades/year target.
WHY IT WORKS BOTH MARKETS: Symmetrical channels work for long breakouts (bull)
and short breakdowns (bear). Volume confirms institutional moves.

TARGET: 50-150 total over 4 years. HARD MAX: 200.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_atr_donchian_kama_vol_v1"
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

def calculate_kama(close, period=10, fast=2, slow=30):
    """Kaufman's Adaptive Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    # Calculate ER (Efficiency Ratio)
    direction = np.abs(close[period:] - close[:-period])
    volatility = np.zeros(n - period)
    for i in range(1, n - period):
        volatility[i] = np.sum(np.abs(close[i+1:i+period+1] - close[i:i+period]))
    
    er = np.zeros(n)
    er[period:] = direction / (volatility + 1e-10)
    er[:period] = 0
    
    # Smoothing constant
    fast_const = 2 / (fast + 1)
    slow_const = 2 / (slow + 1)
    const = er * (fast_const - slow_const) + slow_const
    kama = np.zeros(n)
    kama[0] = close[0]
    
    for i in range(1, n):
        if np.isnan(kama[i-1]) or np.isnan(const[i]):
            kama[i] = close[i]
        else:
            kama[i] = kama[i-1] + const[i] * const[i] * (close[i] - kama[i-1])
    
    return kama

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d KAMA for trend direction
    kama_1d = calculate_kama(df_1d['close'].values, period=10)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    
    # Local 12h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Volume SMA
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # ATR-normalized Donchian channels
    # Use ATR(14) * 3 as the channel width (adaptive to volatility)
    atr_mult_upper = 3.0
    atr_mult_lower = 3.0
    
    # Rolling highest high and lowest low over 20 bars
    rolling_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    rolling_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # ATR-normalized channels
    upper_band = rolling_high + atr_mult_upper * atr_14
    lower_band = rolling_low - atr_mult_lower * atr_14
    mid_band = (upper_band + lower_band) / 2
    
    signals = np.zeros(n)
    SIZE = 0.30  # Slightly higher since fewer trades expected
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_bar = 0
    
    warmup = 60  # Need enough for rolling calculations
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(kama_1d_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === TREND DIRECTION (1d KAMA) ===
        price_above_kama = close[i] > kama_1d_aligned[i]
        trend_strength = abs(close[i] - kama_1d_aligned[i]) / (atr_14[i] + 1e-10)
        
        # === VOLUME CONFIRMATION ===
        vol_confirmed = vol_ratio[i] > 1.5
        
        # === ENTRY SIGNALS ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG: Break above upper band (ATR-adjusted Donchian high)
            # Price exceeds the 20-bar high + 3*ATR with volume
            if close[i] > upper_band[i - 1] and price_above_kama:
                if vol_confirmed:  # Volume confirmation required for entries
                    desired_signal = SIZE
            
            # === SHORT: Break below lower band (ATR-adjusted Donchian low)
            # Price breaks below 20-bar low - 3*ATR without volume (capitulation)
            if close[i] < lower_band[i - 1] and not price_above_kama:
                if vol_confirmed:  # Volume confirmation for shorts too
                    desired_signal = -SIZE
        
        # === STOPLOSS CHECK (2.5 ATR trailing) ===
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
        
        # === HOLDING PERIOD EXIT (min 4 bars = 2 days) ===
        bars_held = i - entry_bar
        
        if in_position and bars_held >= 4:
            # Exit if price reverts to mid band (mean reversion)
            if position_side > 0 and close[i] < mid_band[i]:
                desired_signal = 0.0
            if position_side < 0 and close[i] > mid_band[i]:
                desired_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        # Exit if trend changes (price crosses KAMA after being aligned)
        if in_position and bars_held >= 2:
            if position_side > 0 and close[i] < kama_1d_aligned[i]:
                desired_signal = 0.0
            if position_side < 0 and close[i] > kama_1d_aligned[i]:
                desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New position or flip
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                entry_bar = i
                if position_side > 0:
                    stop_price = high[i] - 2.5 * entry_atr
                else:
                    stop_price = low[i] + 2.5 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_atr = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = desired_signal
    
    return signals