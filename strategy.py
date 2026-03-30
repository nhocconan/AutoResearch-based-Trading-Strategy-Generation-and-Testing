#!/usr/bin/env python3
"""
Experiment #028: 4h Donchian Breakout + Bollinger Bandwidth Squeeze + Volume Confirmation

HYPOTHESIS: Bollinger Band squeeze (low volatility) precedes breakouts. By entering
when volatility compresses AND price breaks Donchian structure, we catch explosive moves.
This works in BOTH bull (breakout rallies) and bear (breakdown crashes) because we
trade the direction of the break with symmetrical rules.

WHY IT WORKS: The squeeze acts as a "coil" that releases energy. Unlike trend indicators
that lag, bandwidth contraction is a leading indicator. Combined with Donchian(20) for
structure and volume for confirmation, this catches institutional moves.

KEY DIFFERENCE from failed strategies: Added BB squeeze filter prevents entering
choppy ranges. Previous Donchian strategies entered on every breakout—now we only
enter when volatility SUPPORTS a move.

TARGET: 60-120 total trades over 4 years = 15-30/year. HARD MAX: 200.
Signal size: 0.25.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_bb_squeeze_vol_v1"
timeframe = "4h"
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

def calculate_bb_width_percentile(close, period=20, lookback=120):
    """Bollinger Bandwidth as percentile of recent history - squeeze detection"""
    n = len(close)
    if n < period + lookback:
        return np.full(n, np.nan)
    
    # BB middle
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    # Bandwidth = (Upper - Lower) / Middle
    upper = sma + 2 * std
    lower = sma - 2 * std
    bandwidth = (upper - lower) / np.where(sma != 0, sma, 1)
    
    # Percentile of bandwidth over lookback period
    percentile = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(lookback, n):
        recent_bw = bandwidth[i - lookback:i + 1]
        current_bw = bandwidth[i]
        # What percentile is current bandwidth among last 120 values?
        percentile[i] = (np.sum(recent_bw <= current_bw) / len(recent_bw)) * 100
    
    return percentile

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d SMA50 for trend direction
    sma_1d = pd.Series(df_1d['close'].values).rolling(window=50, min_periods=50).mean().values
    sma_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_1d)
    
    # Local 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    bb_pct = calculate_bb_width_percentile(close, period=20, lookback=120)
    
    # Donchian channels (20 periods = 3.3 days on 4h)
    donchian_period = 20
    donchian_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # Volume confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
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
    
    warmup = 150  # Need enough for BB percentile(120) + Donchian(20) + buffer
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(sma_1d_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(bb_pct[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === TREND DIRECTION (1d SMA50) ===
        price_above_1d_sma = close[i] > sma_1d_aligned[i]
        
        # === VOLATILITY REGIME (BB Squeeze) ===
        # Squeeze: bandwidth at <20th percentile of last 120 bars
        is_squeeze = bb_pct[i] < 20.0
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.8
        
        # === PRICE STRUCTURE ===
        current_high = high[i]
        current_low = low[i]
        
        # Previous bar's Donchian values (use i-1 to avoid look-ahead)
        prev_donchian_high = donchian_high[i - 1] if i > 0 else 0
        prev_donchian_low = donchian_low[i - 1] if i > 0 else 0
        prev_close = close[i - 1] if i > 0 else close[i]
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG: Breakout with squeeze + volume ===
            # Price closes above previous Donchian high with squeeze confirming
            if prev_close > prev_donchian_high and price_above_1d_sma:
                if is_squeeze and vol_spike:
                    desired_signal = SIZE
                elif is_squeeze and not vol_spike:
                    # Squeeze alone if volume not available but squeeze strong
                    if bb_pct[i] < 10.0:
                        desired_signal = SIZE * 0.5  # Half size without volume
            
            # === SHORT: Breakdown with squeeze + volume ===
            if prev_close < prev_donchian_low and not price_above_1d_sma:
                if is_squeeze and vol_spike:
                    desired_signal = -SIZE
                elif is_squeeze and not vol_spike:
                    if bb_pct[i] < 10.0:
                        desired_signal = -SIZE * 0.5
        
        # === STOPLOSS CHECK (2.0 ATR trailing) ===
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
        
        # === TIME-BASED EXIT (hold at least 4 bars = ~16h) ===
        bars_held = i - entry_bar
        
        if in_position and bars_held >= 4:
            # Exit if price reverts to 20-bar midpoint
            midpoint = (donchian_high[i - 1] + donchian_low[i - 1]) / 2 if i > 0 else close[i]
            if position_side > 0 and close[i] < midpoint:
                desired_signal = 0.0
            if position_side < 0 and close[i] > midpoint:
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
        
        signals[i] = desired_signal
    
    return signals