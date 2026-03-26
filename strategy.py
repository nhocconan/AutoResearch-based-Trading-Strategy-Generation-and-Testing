#!/usr/bin/env python3
"""
Experiment #005: 12h Williams Alligator + Donchian Structure + 1d Trend Filter

HYPOTHESIS: Williams Alligator (3/5/8 SMAs) captures institutional order flow 
through the jaw/teeth/lips alignment. Combined with Donchian breakouts for 
structural boundaries and 1d HMA trend filter, this should work in both bull 
and bear markets.

WHY THIS SHOULD WORK IN BOTH BULL AND BEAR:
- Alligator identifies trend direction and exhaustion via jaw/teeth/lips spread
- Bull: Alligator wakes up (lines converge) + price above = long
- Bear: Alligator wakes up + price below = short  
- Range: Alligator sleeps (lines flat) = no trades
- Donchian confirms structural breakouts, filters false signals
- 1d trend filter prevents fighting the larger trend

TARGET: 75-150 total trades over 4 years (12h = ~2920 bars total)
This means entry trigger rate of ~3-5%, much stricter than current 1550 trades.

DB REFERENCE: mtf_12h_simple_donchian_1w_trend_v1 (104 tr, session best)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_alligator_donchian_1d_v1"
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

def calculate_alligator(high, low, close, jaw_period=13, teeth_period=8, lips_period=5):
    """
    Williams Alligator indicator
    Jaw = SMMA(jaw_period) of median price
    Teeth = SMMA(teeth_period) of median price  
    Lips = SMMA(lips_period) of median price
    
    Alligator sleeps = lines converged/flat (choppy)
    Alligator wakes = lines spread (trending)
    """
    n = len(close)
    median = (high + low) / 2.0
    
    # SMMA (Smoothed Moving Average = EMA with alpha=1/period)
    def smma(series, period):
        result = np.full(len(series), np.nan, dtype=np.float64)
        # First valid value is simple SMA
        for i in range(period - 1, len(series)):
            window = series[i - period + 1:i + 1]
            if not np.any(np.isnan(window)):
                sma = np.mean(window)
                result[i] = sma
        # Apply smoothing
        alpha = 1.0 / period
        for i in range(period, len(series)):
            if np.isnan(result[i]):
                continue
            if not np.isnan(result[i - 1]):
                result[i] = result[i - 1] + alpha * (series[i] - result[i - 1])
        return result
    
    jaw = smma(median, jaw_period)
    teeth = smma(median, teeth_period)
    lips = smma(median, lips_period)
    
    return jaw, teeth, lips

def calculate_donchian(high, low, period=20):
    """Donchian Channel - breakouts of 20-period high/low"""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    mid = (upper + lower) / 2.0
    return upper, lower, mid

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HMA for trend direction
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    jaw, teeth, lips = calculate_alligator(high, low, close, jaw_period=13, teeth_period=8, lips_period=5)
    donchian_upper, donchian_lower, donchian_mid = calculate_donchian(high, low, period=20)
    
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
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Cooldown to prevent overtrading
    bars_since_entry = 0
    cooldown_bars = 8  # Minimum 8 bars (4 days) between entries
    
    warmup = 60
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]):
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
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        bars_since_entry += 1 if not in_position else 0
        
        # === 1d TREND FILTER ===
        price_above_1d_hma = close[i] > hma_1d_aligned[i]
        trend_bullish = price_above_1d_hma
        trend_bearish = not price_above_1d_hma
        
        # === ALLIGATOR CONDITION ===
        # Alligator "waking up" = lips crosses above teeth (bull) or below (bear)
        # Alligator aligned = lips > teeth > jaw (bull) or lips < teeth < jaw (bear)
        lips_above_teeth = lips[i] > teeth[i] if not np.isnan(lips[i]) and not np.isnan(teeth[i]) else False
        teeth_above_jaw = teeth[i] > jaw[i] if not np.isnan(teeth[i]) and not np.isnan(jaw[i]) else False
        
        alligator_bullish = lips_above_teeth and teeth_above_jaw
        alligator_bearish = not lips_above_teeth and not teeth_above_jaw
        
        # Alligator spread (awake vs sleeping)
        alligator_spread = abs(lips[i] - jaw[i]) / atr_14[i] if not np.isnan(lips[i]) and not np.isnan(jaw[i]) else 0.0
        alligator_awake = alligator_spread > 0.3  # Lines spread = trending
        
        # === DONCHIAN BREAKOUT ===
        donchian_broken_up = close[i] > donchian_upper[i] if not np.isnan(donchian_upper[i]) else False
        donchian_broken_down = close[i] < donchian_lower[i] if not np.isnan(donchian_lower[i]) else False
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === ENTRY LOGIC (strict - must have confluence) ===
        desired_signal = 0.0
        
        # LONG: Alligator bullish + price above 1d HMA + Donchian breakout + volume
        # Minimum 3 of 4 conditions, with volume required
        if vol_spike:
            bullish_conditions = sum([
                alligator_bullish,
                trend_bullish,
                donchian_broken_up,
                alligator_awake
            ])
            if bullish_conditions >= 3:
                desired_signal = SIZE
        
        # SHORT: Alligator bearish + price below 1d HMA + Donchian breakdown + volume
        if vol_spike:
            bearish_conditions = sum([
                alligator_bearish,
                trend_bearish,
                donchian_broken_down,
                alligator_awake
            ])
            if bearish_conditions >= 3:
                desired_signal = -SIZE
        
        # === COOLDOWN ===
        if bars_since_entry < cooldown_bars:
            desired_signal = 0.0
        
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
        
        # === TRAILING STOP on profit ===
        if in_position and position_side > 0:
            profit_pct = (highest_since_entry - entry_price) / entry_price
            if profit_pct > 0.03:  # 3% profit - tighten stop
                trailing_stop = highest_since_entry - 1.5 * entry_atr
                stop_price = max(stop_price, trailing_stop)
                if low[i] < stop_price:
                    stoploss_triggered = True
                    desired_signal = 0.0
        
        if in_position and position_side < 0:
            profit_pct = (entry_price - lowest_since_entry) / entry_price
            if profit_pct > 0.03:
                trailing_stop = lowest_since_entry + 1.5 * entry_atr
                stop_price = min(stop_price, trailing_stop)
                if high[i] > stop_price:
                    stoploss_triggered = True
                    desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                bars_since_entry = 0
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