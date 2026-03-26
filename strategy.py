#!/usr/bin/env python3
"""
Experiment #008: 12h Primary + 1w HTF — Donchian Breakout with Volume

Hypothesis: 12h timeframe with weekly trend bias creates optimal trade frequency
(50-150 over 4 years = 12-37/year). One clear signal type (Donchian breakout) with
volume confirmation and weekly HMA trend filter.

Why it should work in BOTH bull AND bear:
- Bull: Price breaks above Donchian upper + weekly trend up = momentum continuation
- Bear: Price breaks below Donchian lower + weekly trend down = momentum continuation
- Range: Wider Donchian(16) = fewer breakouts = avoids whipsaw
- Weekly HMA(21) filter = direction bias that adapts to regime

Key design (learned from 16,000+ experiments):
1. ONE signal type: Donchian breakout (proven pattern)
2. Weekly HMA trend filter (avoids fighting major trend)
3. Volume confirmation (filters false breakouts)
4. Tight entry: requires breakout + trend + volume ALL aligned
5. ATR-based stoploss (signal→0 when price moves 2.5x ATR against position)

Why this beats failed strategies:
- NO Fisher/RSI/KAMA stacking (caused overtrading in #013-#019)
- NO "loose" thresholds (caused 400-17000+ trades)
- Single clear signal with confluence = controlled trade frequency

Target: Sharpe>0.5, trades=50-150 total, DD>-40%
Timeframe: 12h
Size: 0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_volume_1w_v1"
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

def calculate_donchian(high, low, period=16):
    """Donchian Channel - price channel breakout signal"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = np.full(n, np.nan, dtype=np.float64)
    lower = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower

def calculate_volume_confirmation(volume, period=20):
    """Volume SMA for confirmation - requires volume > 20-period average"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_sma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return vol_sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load 1w HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align weekly indicators
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    donch_upper, donch_lower = calculate_donchian(high, low, period=16)
    vol_sma_20 = calculate_volume_confirmation(volume, period=20)
    
    signals = np.zeros(n)
    SIZE = 0.30  # Discrete position size
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup period (12h bars need more warmup for weekly alignment)
    min_bars = max(50, 16)  # Donchian period + buffer
    
    for i in range(min_bars, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
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
        
        if np.isnan(vol_sma_20[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === WEEKLY TREND FILTER ===
        price_above_1w = close[i] > hma_1w_aligned[i]
        price_below_1w = close[i] < hma_1w_aligned[i]
        
        # === VOLUME CONFIRMATION ===
        volume_confirmed = volume[i] > vol_sma_20[i]
        
        # === DONCHIAN BREAKOUT (tight: require ALL 3 conditions) ===
        # Need previous bar for proper breakout detection (no look-ahead)
        if i > 0:
            donch_upper_prev = donch_upper[i-1] if not np.isnan(donch_upper[i-1]) else 0
            donch_lower_prev = donch_lower[i-1] if not np.isnan(donch_lower[i-1]) else 0
        else:
            signals[i] = 0.0
            continue
        
        # Long breakout: price CLOSES above previous upper channel
        breakout_long = close[i] > donch_upper_prev
        
        # Short breakout: price CLOSES below previous lower channel
        breakout_short = close[i] < donch_lower_prev
        
        # === ENTRY LOGIC (TIGHT: requires breakout + trend + volume) ===
        desired_signal = 0.0
        
        # LONG: Weekly trend up + Donchian breakout + Volume confirmation
        if breakout_long and price_above_1w and volume_confirmed:
            desired_signal = SIZE
        
        # SHORT: Weekly trend down + Donchian breakout + Volume confirmation
        elif breakout_short and price_below_1w and volume_confirmed:
            desired_signal = -SIZE
        
        # === STOPLOSS CHECK (2.5x ATR trailing) ===
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
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE * 0.9:
            final_signal = SIZE
        elif desired_signal <= -SIZE * 0.9:
            final_signal = -SIZE
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                in_position = True
                position_side = int(np.sign(final_signal))
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
        
        signals[i] = final_signal
    
    return signals