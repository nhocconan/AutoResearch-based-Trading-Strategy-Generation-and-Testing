#!/usr/bin/env python3
"""
Experiment #003: 4h Donchian Breakout + 12h HMA Trend + Volume + Choppiness

HYPOTHESIS: Donchian(20) breakouts capture momentum moves, but only when:
1. 12h HMA confirms trend direction (avoid counter-trend breakouts that fail)
2. Volume spike confirms institutional participation (>1.5x 20-avg)
3. Choppiness < 55 ensures we're in trending regime (not choppy whipsaw)

WHY THIS SHOULD WORK IN BOTH BULL AND BEAR:
- Bull: Long breakouts above Donchian high when 12h HMA sloping up
- Bear: Short breakouts below Donchian low when 12h HMA sloping down
- Range: Choppiness filter blocks entries (CHOP > 55 = no trades)

TARGET: 100-150 total trades over 4 years (25-37/year) - sweet spot for 4h
DB reference: mtf_4h_hma_donchian_volume_rsi_12h_atr_v1 (SOL Sharpe=1.382, 95 trades)

KEY DESIGN:
1. Donchian(20) breakout as entry trigger
2. 12h HMA slope for trend bias (only trade with HTF trend)
3. Volume spike >1.5x confirmation
4. Choppiness < 55 regime filter
5. ATR(14) 2.5x stoploss
6. Signal: ±0.25 (discrete, conservative sizing)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_hma12_vol_chop_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - faster response than EMA"""
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

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures market choppiness
    CHOP > 61.8 = ranging, CHOP < 38.2 = trending
    We use < 55 as threshold to allow some neutral periods
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

def calculate_donchian(high, low, period=20):
    """Donchian Channel - highest high and lowest low over period"""
    n = len(high)
    upper = np.full(n, np.nan, dtype=np.float64)
    lower = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower

def calculate_ema(close, span):
    """Exponential Moving Average"""
    return pd.Series(close).ewm(span=span, min_periods=span, adjust=False).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load 12h data for trend filter - CALL ONCE BEFORE LOOP
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h HMA for trend direction
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate 12h HMA slope (current vs 3 bars ago on 12h)
    hma_12h_slope = np.full(n, np.nan, dtype=np.float64)
    for i in range(3, n):
        if not np.isnan(hma_12h_aligned[i]) and not np.isnan(hma_12h_aligned[i-3]):
            hma_12h_slope[i] = hma_12h_aligned[i] - hma_12h_aligned[i-3]
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    donch_upper, donch_lower = calculate_donchian(high, low, period=20)
    
    # Volume moving average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    signals = np.zeros(n)
    SIZE = 0.25  # Conservative position size
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup for all indicators
    warmup = 60
    
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
        
        if np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_12h_aligned[i]) or np.isnan(hma_12h_slope[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME CHECK ===
        chop = chop_14[i]
        is_trending = chop < 55.0  # Allow trending and neutral, block choppy
        
        # === TREND BIAS (12h HMA slope) ===
        hma_sloping_up = hma_12h_slope[i] > 0
        hma_sloping_down = hma_12h_slope[i] < 0
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === DONCHIAN BREAKOUT DETECTION ===
        # Breakout = close crosses above upper or below lower
        prev_upper = donch_upper[i-1] if i > 0 else donch_upper[i]
        prev_lower = donch_lower[i-1] if i > 0 else donch_lower[i]
        
        breakout_long = close[i] > prev_upper and close[i-1] <= prev_upper
        breakout_short = close[i] < prev_lower and close[i-1] >= prev_lower
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # LONG: Donchian breakout + 12h HMA up + volume OR trending regime
        if is_trending and breakout_long:
            if hma_sloping_up:
                if vol_spike:
                    desired_signal = SIZE
                elif vol_ratio[i] > 1.2:
                    desired_signal = SIZE
        
        # SHORT: Donchian breakout + 12h HMA down + volume OR trending regime
        if is_trending and breakout_short:
            if hma_sloping_down:
                if vol_spike:
                    desired_signal = -SIZE
                elif vol_ratio[i] > 1.2:
                    desired_signal = -SIZE
        
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
        
        # === TAKE PROFIT at opposite Donchian band ===
        tp_triggered = False
        if in_position and position_side > 0:
            # Exit if price reaches upper band + 1 ATR (extended move)
            if high[i] >= donch_upper[i] + 0.5 * atr_14[i]:
                tp_triggered = True
        
        if in_position and position_side < 0:
            # Exit if price reaches lower band - 1 ATR (extended move)
            if low[i] <= donch_lower[i] - 0.5 * atr_14[i]:
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