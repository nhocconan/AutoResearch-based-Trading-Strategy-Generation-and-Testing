#!/usr/bin/env python3
"""
Experiment #026: 4h Supertrend + 1d HMA Trend Filter + Volume Confirmation

HYPOTHESIS: Supertrend provides reliable trend reversal signals on 4h.
Combined with 1d HMA trend alignment (filtering counter-trend trades) and 
volume confirmation, this captures institutional moves while avoiding 
whipsaws in both bull and bear markets.

WHY IT SHOULD WORK:
- Supertrend is a proven trend-following indicator with built-in stoploss
- 1d HMA filters out trades against the larger trend (reduces losing trades)
- Volume confirmation ensures institutional participation
- 4h timeframe = ~45K bars over 4 years, ~100-150 trades expected
- Works in bull (longs with trend) and bear (shorts when price < 1d HMA)

TIMEFRAME: 4h primary
HTF: 1d for trend filter
TARGET: 75-200 total trades over 4 years (19-50/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_supertrend_1d_hma_vol_v1"
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

def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """
    Supertrend Indicator
    Returns: supertrend values (positive = bullish, negative = bearish)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    # ATR calculation
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], 
                    abs(high[i] - close[i-1]), 
                    abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # HL2 (Typical Price)
    hl2 = (high + low) / 2.0
    
    # Upper and Lower bands
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    supertrend = np.zeros(n, dtype=np.float64)
    trend = np.zeros(n, dtype=np.int8)  # 1 = uptrend, -1 = downtrend
    
    for i in range(n):
        if i == 0:
            supertrend[i] = lower_band[i]
            trend[i] = 1
        else:
            # Previous values
            prev_close = close[i-1]
            prev_st = supertrend[i-1]
            prev_trend = trend[i-1]
            
            if pd.isna(prev_st):
                supertrend[i] = lower_band[i]
                trend[i] = 1
            else:
                if prev_trend == 1:  # Uptrend
                    if close[i] < prev_st:
                        trend[i] = -1
                        supertrend[i] = upper_band[i]
                    else:
                        trend[i] = 1
                        supertrend[i] = max(lower_band[i], prev_st)
                else:  # Downtrend
                    if close[i] > prev_st:
                        trend[i] = 1
                        supertrend[i] = lower_band[i]
                    else:
                        trend[i] = -1
                        supertrend[i] = min(upper_band[i], prev_st)
    
    return supertrend, trend

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
    
    # === Load HTF data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d HMA for trend filter
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # === Calculate local 4h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Supertrend: period=10, multiplier=3
    supertrend_vals, trend_dir = calculate_supertrend(high, low, close, period=10, multiplier=3.0)
    
    # Volume MA for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    signals = np.zeros(n)
    SIZE = 0.25  # Conservative sizing
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    prev_trend = 0  # Track previous trend direction for reversal detection
    
    warmup = 50
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(supertrend_vals[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            prev_trend = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            prev_trend = trend_dir[i] if not pd.isna(trend_dir[i]) else 0
            continue
        
        current_trend = trend_dir[i]
        
        # === TREND FILTER (1d HMA) ===
        price_above_1d_hma = close[i] > hma_1d_aligned[i]
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === SUPERTREND REVERSAL DETECTION ===
        # Trend changed from down to up = bullish reversal
        trend_reversal_up = (prev_trend == -1) and (current_trend == 1)
        # Trend changed from up to down = bearish reversal
        trend_reversal_down = (prev_trend == 1) and (current_trend == -1)
        
        desired_signal = 0.0
        
        # === STOPLOSS CHECK (3 ATR trailing) ===
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
        
        # === ENTRY LOGIC ===
        if not in_position and not stoploss_triggered:
            # === NEW LONG ENTRY ===
            # Supertrend reversal to up + price above 1d HMA + volume
            if trend_reversal_up:
                if price_above_1d_hma:
                    desired_signal = SIZE
                elif vol_spike:  # Also allow if strong volume even against trend
                    desired_signal = SIZE * 0.5  # Half size for counter-trend
            
            # === NEW SHORT ENTRY ===
            # Supertrend reversal to down + price below 1d HMA
            if trend_reversal_down:
                if not price_above_1d_hma:
                    desired_signal = -SIZE
                elif vol_spike:  # Also allow if strong volume even against trend
                    desired_signal = -SIZE * 0.5
        
        # === EXIT LOGIC ===
        exit_triggered = False
        
        if in_position:
            # Exit on opposite trend reversal (protected profit)
            if position_side > 0 and trend_reversal_down:
                exit_triggered = True
            if position_side < 0 and trend_reversal_up:
                exit_triggered = True
            
            # Exit on Supertrend level breach (stop loss)
            if position_side > 0 and low[i] < supertrend_vals[i]:
                exit_triggered = True
            if position_side < 0 and high[i] > supertrend_vals[i]:
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
        
        # Update trend for next iteration
        prev_trend = current_trend
        
        signals[i] = desired_signal
    
    return signals