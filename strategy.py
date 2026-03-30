#!/usr/bin/env python3
"""
Experiment #028: 12h Donchian Breakout + 1w EMA Trend + ATR Volume Filter

HYPOTHESIS: Donchian(20) on 12h = 10-day channel capturing medium-term swings.
Combined with 1w EMA(21) for trend direction (bullish when price > EMA, bearish otherwise),
and ATR-based volume confirmation to filter false breakouts.

WHY IT WORKS: Simple price-channel breakout is a proven institutional pattern.
Using weekly EMA for trend direction keeps us long in bull markets (2021, 2023-2024)
and short in bear markets (2022). ATR spike confirms institutional participation.
12h is slow enough to avoid fee drag, fast enough to capture meaningful moves.

TARGET: 75-150 total trades over 4 years (19-37/year).
Signal size: 0.30.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_1w_ema_atr_vol_v1"
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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE ===
    df_1w = get_htf_data(prices, '1w')
    
    # 1w EMA(21) for trend direction
    ema_1w = pd.Series(df_1w['close'].values).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Local 12h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # ATR ratio for volume confirmation (current ATR vs 30d average)
    atr_30 = calculate_atr(high, low, close, period=30)
    atr_ratio = atr_14 / np.where(atr_30 > 0, atr_30, 1)
    
    # Donchian channels (20 periods = 10 days on 12h)
    donchian_period = 20
    donchian_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
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
    entry_bar = 0
    
    warmup = 50  # Need enough for indicators
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(ema_1w_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === TREND DIRECTION (1w EMA21) ===
        price_above_1w_ema = close[i] > ema_1w_aligned[i]
        
        # === DONCHIAN BREAKOUT SIGNALS ===
        # Previous bar's Donchian values (use i-1 to avoid look-ahead)
        prev_donchian_high = donchian_high[i - 1] if i > 0 else 0
        prev_donchian_low = donchian_low[i - 1] if i > 0 else 0
        
        # Volume confirmation via ATR spike (volatility expansion = institutional)
        vol_expansion = atr_ratio[i] > 1.5
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG: Breakout above Donchian high with trend confirmation ===
            if high[i] > prev_donchian_high and price_above_1w_ema:
                # Only enter if ATR expanding (confirms move)
                if vol_expansion:
                    desired_signal = SIZE
            
            # === SHORT: Breakdown below Donchian low with trend confirmation ===
            if low[i] < prev_donchian_low and not price_above_1w_ema:
                if vol_expansion:
                    desired_signal = -SIZE
        
        # === STOPLOSS CHECK (2.0 ATR trailing) ===
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
        
        # === HOLDING PERIOD FILTER (min 4 bars = 2 days) ===
        bars_held = i - entry_bar
        if bars_held < 4 and in_position:
            # Don't exit early, but don't enter if signal flips
            if desired_signal != 0 and np.sign(desired_signal) != position_side:
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
                entry_bar = i
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