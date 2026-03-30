#!/usr/bin/env python3
"""
Experiment #008: Williams Alligator + Elder Force Index + 1w Trend

HYPOTHESIS: Williams Alligator identifies institutional accumulation/distribution
(when jaw/teeth/lips compress = smart money building positions). Elder Force Index
confirms if the resulting move has real volume backing. Combined with 1w SMA200
trend filter, this catches major trend changes in both bull and bear.

Key insight: Alligator compression BEFORE breakout = accumulator pattern.
Force Index crossing zero AFTER compression = confirmation of direction.

WHY IT WORKS: Simple 2-condition entry (alignment + confirmation). No multiple
filters that kill trades. Target 75-150 total trades over 4 years.

timeframe: 12h
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_alligator_force_1w_v1"
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

def williams_alligator(high, low, close):
    """
    Williams Alligator: 3 smoothed moving averages
    Jaw: SMMA of close, period 13, offset 8
    Teeth: SMMA of close, period 8, offset 5  
    Lips: SMMA of close, period 5, offset 3
    
    Returns: jaw, teeth, lips
    """
    n = len(close)
    
    # SMMA (Smoothed Moving Average) = EMA with alpha = 1/period
    def smma(series, period):
        result = np.zeros(n)
        result[0] = series[0]
        alpha = 1.0 / period
        for i in range(1, n):
            result[i] = (result[i-1] * (1 - alpha) + series[i] * alpha)
        return result
    
    jaw = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    return jaw, teeth, lips

def elder_force_index(close, volume, period=13):
    """
    Elder Force Index: (close - prev_close) * volume
    Uses EMA smoothing. Positive = bullish force, Negative = bearish force.
    """
    n = len(close)
    efi = np.zeros(n)
    efi[0] = 0.0
    
    for i in range(1, n):
        efi[i] = (close[i] - close[i-1]) * volume[i]
    
    # Smooth with EMA
    efi_ema = pd.Series(efi).ewm(span=period, min_periods=period, adjust=False).mean().values
    return efi_ema

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1w = get_htf_data(prices, '1w')
    
    # 1w SMA200 for trend (works on all markets)
    sma_1w = pd.Series(df_1w['close'].values).rolling(window=200, min_periods=200).mean().values
    sma_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_1w)
    
    # === Local 12h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Williams Alligator
    jaw, teeth, lips = williams_alligator(high, low, close)
    
    # Elder Force Index
    efi = elder_force_index(close, volume, period=13)
    
    # Volume ratio for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Signals
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_bar = 0
    
    warmup = 300  # Need enough for 1w SMA200 alignment buffer
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(sma_1w_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === 1w TREND FILTER ===
        price_above_1w_sma = close[i] > sma_1w_aligned[i]
        price_below_1w_sma = close[i] < sma_1w_aligned[i]
        
        # === ALLIGATOR ALIGNMENT ===
        # Bullish: lips > teeth > jaw (all aligned upward)
        alligator_bull = (lips[i] > teeth[i]) and (teeth[i] > jaw[i]) and (jaw[i] > jaw[i-1])
        # Bearish: lips < teeth < jaw (all aligned downward)
        alligator_bear = (lips[i] < teeth[i]) and (teeth[i] < jaw[i]) and (jaw[i] < jaw[i-1])
        
        # Alligator compression (potential breakout setup)
        jaw_range = abs(jaw[i] - jaw[i-3]) / jaw[i] if jaw[i] > 0 else 0
        teeth_range = abs(teeth[i] - teeth[i-3]) / teeth[i] if teeth[i] > 0 else 0
        lips_range = abs(lips[i] - lips[i-3]) / lips[i] if lips[i] > 0 else 0
        is_compressed = (jaw_range < 0.005) and (teeth_range < 0.005) and (lips_range < 0.005)
        
        # === FORCE INDEX ===
        efi_positive = efi[i] > 0
        efi_negative = efi[i] < 0
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG ENTRY ===
            # Trend up + Alligator aligned + Force positive + (compressed OR volume)
            if price_above_1w_sma and alligator_bull and efi_positive:
                if is_compressed or vol_ratio[i] > 1.2:
                    desired_signal = SIZE
            
            # === SHORT ENTRY ===
            # Trend down + Alligator aligned + Force negative + (compressed OR volume)
            if price_below_1w_sma and alligator_bear and efi_negative:
                if is_compressed or vol_ratio[i] > 1.2:
                    desired_signal = -SIZE
        
        # === STOPLOSS (2.5 ATR) ===
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                desired_signal = 0.0
        
        # === MINIMUM HOLD (3 bars = 1.5 days on 12h to reduce churn) ===
        bars_held = i - entry_bar
        if in_position and bars_held >= 3:
            # Take profit on opposite Force signal
            if position_side > 0 and efi_negative:
                desired_signal = 0.0
            if position_side < 0 and efi_positive:
                desired_signal = 0.0
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New position or flip
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                entry_bar = i
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                stop_price = 0.0
        
        signals[i] = desired_signal
    
    return signals