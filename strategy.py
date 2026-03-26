#!/usr/bin/env python3
"""
Experiment #005: 12h Primary + 1d HTF — Simple Donchian Breakout

HYPOTHESIS:
12h Donchian(20) breakout with 1d trend bias and volume confirmation is the optimal
balance between trade frequency and signal quality. This mirrors the proven pattern
from DB that achieved SOLUSDT test Sharpe 1.10-1.38. Simple = fewer trades = less fee drag.

Key design choices:
1. ONLY 3 conditions per entry (Donchian break + 1d bias + volume spike)
2. 12h TF = ~365 bars/year = 25-40 trades/year target (within 12-37/year range)
3. 1d SMA for trend direction (bull/bear filter)
4. Volume > 20d MA as confirmation (prevents false breakouts)
5. ATR 2.5x trailing stoploss for risk management
6. Discrete sizes: 0.25/0.30 (never exceed 0.30)

Why 12h over 4h:
- 4h = ~1820 bars/4yr → tends to overtrade (>300 trades) → fee drag kills performance
- 12h = ~730 bars/4yr → naturally constrained by structure → target 75-150 trades
- DB shows 12h keep rate = 54%, much better than 4h average

Why this should work in BOTH bull and bear:
- Bull: Donchian breakout above 1d SMA = strong momentum continuation
- Bear: Donchian breakdown below 1d SMA = continuation of downtrend
- Volume spike confirms institutional participation (not noise)
- Choppiness filter avoids ranging markets where breakouts fail

Target: Sharpe > 0.5, 75-150 train trades, DD > -40%
Timeframe: 12h | Size: 0.25-0.30 | Leverage: 1.0
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_volume_1d_simple_v1"
timeframe = "12h"
leverage = 1.0

def calculate_sma(close, period):
    """Simple Moving Average with min_periods"""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

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
    """Donchian Channel - returns upper (resistance), lower (support)"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = np.full(n, np.nan, dtype=np.float64)
    lower = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower

def calculate_volume_ma(volume, period=20):
    """Volume Moving Average"""
    return pd.Series(volume).rolling(window=period, min_periods=period).mean().values

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - filter out ranging markets
    CHOP > 61.8 = ranging (skip), CHOP < 38.2 = trending (trade)
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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d SMA (21 period) and align to 12h
    sma_1d_raw = calculate_sma(df_1d['close'].values, period=21)
    sma_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_1d_raw)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    donch_upper, donch_lower = calculate_donchian(high, low, period=20)
    volume_ma20 = calculate_volume_ma(volume, period=20)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup period (need 20 bars for Donchian + 20 for volume MA)
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
        
        if np.isnan(volume_ma20[i]) or np.isnan(chop_14[i]):
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
        
        # === REGIME FILTER ===
        # Only trade in trending or neutral markets, skip ranging
        chop = chop_14[i]
        skip_range = chop > 61.8
        
        # === TREND DIRECTION (1d SMA bias) ===
        price_above_1d = close[i] > sma_1d_aligned[i]
        price_below_1d = close[i] < sma_1d_aligned[i]
        
        # === DONCHIAN BREAKOUT (price crosses previous channel) ===
        donch_breakout_long = False
        donch_breakout_short = False
        
        if i > 0 and not np.isnan(donch_upper[i-1]) and not np.isnan(donch_lower[i-1]):
            # Breakout = close above previous 20-bar high
            donch_breakout_long = close[i] > donch_upper[i-1]
            # Breakdown = close below previous 20-bar low
            donch_breakout_short = close[i] < donch_lower[i-1]
        
        # === VOLUME CONFIRMATION ===
        volume_spike = volume[i] > volume_ma20[i] * 1.1  # 10% above average
        
        # === ENTRY LOGIC (3 conditions each: breakout + trend + volume) ===
        desired_signal = 0.0
        
        if not skip_range:
            # LONG: Breakout above 12h Donchian + price above 1d SMA + volume confirming
            if donch_breakout_long and price_above_1d and volume_spike:
                desired_signal = SIZE_STRONG
            
            # SHORT: Breakdown below 12h Donchian + price below 1d SMA + volume confirming
            elif donch_breakout_short and price_below_1d and volume_spike:
                desired_signal = -SIZE_STRONG
        
        # === TRAILING STOPLOSS (2.5x ATR) ===
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
        if desired_signal >= SIZE_STRONG * 0.9:
            final_signal = SIZE_STRONG
        elif desired_signal <= -SIZE_STRONG * 0.9:
            final_signal = -SIZE_STRONG
        elif desired_signal >= SIZE_BASE * 0.9:
            final_signal = SIZE_BASE
        elif desired_signal <= -SIZE_BASE * 0.9:
            final_signal = -SIZE_BASE
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                # New position or reversal
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