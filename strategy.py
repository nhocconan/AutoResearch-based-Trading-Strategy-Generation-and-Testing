#!/usr/bin/env python3
"""
Experiment #008: 12h Donchian Breakout + Volume + Weekly Trend + Chop Filter

HYPOTHESIS: Donchian channel breakouts capture momentum moves, but only work when:
1. Weekly trend confirms direction (avoid counter-trend breakouts that fail)
2. Volume spike confirms institutional participation (not fake breakout)
3. Market is trending, not choppy (Choppiness Index < 50)

WHY THIS SHOULD WORK IN BOTH BULL AND BEAR:
- Bull: Long breakouts above Donchian high when 1w HMA bullish
- Bear: Short breakouts below Donchian low when 1w HMA bearish
- Range: Choppiness filter blocks entries (CHOP > 55 = no trades)
- Volume confirmation filters out low-liquidity fake breakouts

TARGET: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.
This is TIGHTER than 4h strategies because 12h has fewer bars.

KEY DESIGN (tight entry to avoid overtrading):
1. 1w HMA(21) for major trend bias (call ONCE before loop)
2. Donchian(20) breakout structure (20-bar high/low)
3. Volume spike > 1.5x 20-avg (institutional confirmation)
4. Choppiness < 50 (trending regime only)
5. ATR(14) trailing stop at 2.5x (tight risk management)
6. Signal: discrete 0.0, ±0.25, ±0.30 (minimize churn)

CRITICAL: All conditions must align = fewer trades = less fee drag.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_vol_chop_1w_v1"
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

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures market choppiness
    CHOP > 61.8 = ranging (no trades), CHOP < 38.2 = strongly trending
    We use CHOP < 50 as threshold to allow trades
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
    
    # === LOAD 1W DATA ONCE BEFORE LOOP ===
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w HMA for trend bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # === CALCULATE 12H INDICATORS ===
    atr_14 = calculate_atr(high, low, close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
    # Volume moving average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    signals = np.zeros(n)
    SIZE = 0.25  # Conservative size for 12h
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup for all indicators
    warmup = 100
    
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
        
        if np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME CHECK ===
        chop = chop_14[i]
        is_trending = chop < 50.0  # Only trade in trending markets
        
        # === TREND BIAS (1w HMA) ===
        price_above_1w_hma = close[i] > hma_1w_aligned[i]
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === DONCHIAN BREAKOUT DETECTION ===
        prev_upper = donchian_upper[i-1] if i > 0 else donchian_upper[i]
        prev_lower = donchian_lower[i-1] if i > 0 else donchian_lower[i]
        
        # Breakout above Donchian upper
        breakout_long = close[i] > prev_upper and close[i-1] <= prev_upper
        
        # Breakout below Donchian lower
        breakout_short = close[i] < prev_lower and close[i-1] >= prev_lower
        
        # === ENTRY LOGIC (ALL CONDITIONS MUST ALIGN) ===
        desired_signal = 0.0
        
        # LONG: Donchian breakout + bullish 1w trend + volume spike + trending regime
        if breakout_long and price_above_1w_hma and vol_spike and is_trending:
            desired_signal = SIZE
        
        # SHORT: Donchian breakout + bearish 1w trend + volume spike + trending regime
        if breakout_short and not price_above_1w_hma and vol_spike and is_trending:
            desired_signal = -SIZE
        
        # === STOPLOSS CHECK (ATR trailing stop) ===
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
        
        # === TAKE PROFIT (opposite Donchian band) ===
        tp_triggered = False
        if in_position and position_side > 0:
            # TP at Donchian lower (mean reversion after extended move)
            if not np.isnan(donchian_lower[i]) and low[i] <= donchian_lower[i]:
                tp_triggered = True
            # Or 3R profit
            if high[i] >= entry_price + 3.0 * entry_atr:
                tp_triggered = True
        
        if in_position and position_side < 0:
            # TP at Donchian upper
            if not np.isnan(donchian_upper[i]) and high[i] >= donchian_upper[i]:
                tp_triggered = True
            # Or 3R profit
            if low[i] <= entry_price - 3.0 * entry_atr:
                tp_triggered = True
        
        if tp_triggered:
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