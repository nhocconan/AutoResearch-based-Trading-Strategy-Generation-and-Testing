#!/usr/bin/env python3
"""
Experiment #003-2: 4h Donchian + 1d HMA + Volume Confirmation (TIGHT)

HYPOTHESIS: Tighter entry conditions (not loose!) = fewer trades = less fee drag.
The key insight from 16,000+ experiments: overtrading kills strategies. The DB's
best performers (Sharpe 1.3-1.7) have 75-300 STRICT trades, not 500+ loose trades.

Key design choices:
1. STRICT: Only enter on Donchian breakout + 1d HMA alignment + volume spike
2. 1d HMA for trend (proven pattern from mtf_4h_hma_donchian_volume_rsi_12h_atr_v1)
3. Donchian(20) for structure (proven in DB winners)
4. Volume spike confirmation (prevents false breakouts)
5. Choppiness filter (>61 = skip, avoid whipsaws)
6. Discrete signals: 0.30 only (no half positions = fewer changes = less fees)
7. 2x ATR stoploss (tight = protects capital)

Why this should work in both bull AND bear:
- Long entries: only when 1d HMA rising AND price breaks 4h Donchian high
- Short entries: only when 1d HMA falling AND price breaks 4h Donchian low
- Bear 2022: choppiness filter skips ranging periods, only trades breaks
- Bull 2021/2023: HMA rising = strong long bias catches rallies

Target: Sharpe>0.8, trades 75-200 total over 4 years (strict!), DD>-30%
Timeframe: 4h with 1d HTF reference
Size: 0.30 discrete only
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_hma_volume_tight_v1"
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
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_donchian(high, low, period=20):
    """Donchian Channel - price channel breakout structure"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = np.full(n, np.nan, dtype=np.float64)
    lower = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index - filter for trending vs ranging"""
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

def calculate_volume_ratio(volume, period=20):
    """Volume ratio vs average - confirms breakouts"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_ma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    ratio = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        if vol_ma[i] > 0:
            ratio[i] = volume[i] / vol_ma[i]
    
    return ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load 1d HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HMA(21) for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 1d HMA for slope (trend direction)
    hma_1d_slope_raw = calculate_hma(df_1d['close'].values, period=8)
    hma_1d_slope_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_slope_raw)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    donch_upper, donch_lower = calculate_donchian(high, low, period=20)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    vol_ratio = calculate_volume_ratio(volume, period=20)
    
    signals = np.zeros(n)
    SIZE = 0.30  # Discrete: only 0.30 or 0.0
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup
    min_bars = 50
    
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
        
        if np.isnan(chop_14[i]) or np.isnan(vol_ratio[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1d_slope_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # Skip if very choppy (avoid whipsaws in 2022 bear)
        is_choppy = chop_14[i] > 61.8
        
        # 1d trend bias
        price_above_1d = close[i] > hma_1d_aligned[i]
        hma_1d_rising = hma_1d_slope_aligned[i] > hma_1d_slope_aligned[i-1] if i > 0 and not np.isnan(hma_1d_slope_aligned[i-1]) else False
        hma_1d_falling = hma_1d_slope_aligned[i] < hma_1d_slope_aligned[i-1] if i > 0 and not np.isnan(hma_1d_slope_aligned[i-1]) else False
        
        # 1d bullish: price above HMA AND HMA rising
        is_1d_bullish = price_above_1d and hma_1d_rising
        is_1d_bearish = (not price_above_1d) and hma_1d_falling
        
        # Volume confirmation: need 1.5x average volume on breakout
        has_volume = vol_ratio[i] >= 1.5
        
        # Donchian breakout detection
        donch_breakout_long = False
        donch_breakout_short = False
        
        if i > 0 and not np.isnan(donch_upper[i-1]) and not np.isnan(donch_lower[i-1]):
            donch_breakout_long = close[i] > donch_upper[i-1] and close[i-1] <= donch_upper[i-1]
            donch_breakout_short = close[i] < donch_lower[i-1] and close[i-1] >= donch_lower[i-1]
        
        # === ENTRY LOGIC (TIGHT - only high probability setups) ===
        desired_signal = 0.0
        
        # LONG: 1d bullish + 4h Donchian breakout + volume spike
        # Skip if choppy (avoid false breakouts in range)
        if not is_choppy and is_1d_bullish and donch_breakout_long and has_volume:
            desired_signal = SIZE
        
        # SHORT: 1d bearish + 4h Donchian breakout + volume spike
        # Skip if choppy
        if not is_choppy and is_1d_bearish and donch_breakout_short and has_volume:
            desired_signal = -SIZE
        
        # === STOPLOSS CHECK (2x ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.0 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.0 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === EXIT: Donchian reversal (opposite breakout) ===
        # Exit long if price breaks below 4h Donchian low
        if in_position and position_side > 0:
            if i > 0 and not np.isnan(donch_lower[i-1]):
                if close[i] < donch_lower[i-1]:
                    desired_signal = 0.0
        
        # Exit short if price breaks above 4h Donchian high
        if in_position and position_side < 0:
            if i > 0 and not np.isnan(donch_upper[i-1]):
                if close[i] > donch_upper[i-1]:
                    desired_signal = 0.0
        
        # === DISCRETIZE (only SIZE or 0) ===
        if abs(desired_signal) >= SIZE * 0.8:
            final_signal = SIZE if desired_signal > 0 else -SIZE
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
                    stop_price = entry_price - 2.0 * entry_atr
                else:
                    stop_price = entry_price + 2.0 * entry_atr
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