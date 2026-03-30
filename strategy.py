#!/usr/bin/env python3
"""
Experiment #027: 12h Camarilla Pivot + 1w Trend + Choppiness Regime

HYPOTHESIS: Camarilla pivot levels represent institutional support/resistance zones.
By combining Camarilla touch signals on 12h with 1w SMA200 for trend direction
and Choppiness Index for regime filtering, we capture mean-reversion trades at 
key levels with institutional confirmation.

WHY 12h: 50-150 target trades over 4 years. Slower than 6h/4h = fewer but
higher-quality signals. 12h Camarilla = 4x daily levels = major institutional zones.

WHY IT WORKS IN BULL AND BEAR: Camarilla is symmetrical (S3-S5 vs R3-R5).
In bull: price respects S3-S5 as support on pullbacks.
In bear: price respects R3-R5 as resistance on rallies.
Choppiness keeps us out of whipsaws.

ENTRY RULES (STRICT - 2-3 conditions only):
- Long: price touches S3/S4 with 1w SMA200 rising AND not choppy
- Short: price touches R3/R4 with 1w SMA200 falling AND not choppy
- Volume confirmation required

TARGET: 50-150 total trades over 4 years = 12-37/year. HARD MAX: 200.
Signal size: 0.30
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_camarilla_chop_1w_v1"
timeframe = "12h"
leverage = 1.0

def calculate_camarilla(high, low, close, period=24):
    """Camarilla pivot levels - institutional support/resistance"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    pivots = np.full((n, 6), np.nan)  # S1-S5, R1-R5
    
    for i in range(period, n):
        h = high[i]
        l = low[i]
        c = close[i]
        rng = h - l
        
        # Classic Camarilla
        pivots[i, 0] = c - rng * 1.1 / 12    # S1
        pivots[i, 1] = c - rng * 1.1 / 6     # S2
        pivots[i, 2] = c - rng * 1.1 / 4     # S3
        pivots[i, 3] = c - rng * 1.1 / 3     # S4
        pivots[i, 4] = c - rng * 1.1 / 2     # S5 (extreme)
        
        pivots[i, 5] = c + rng * 1.1 / 12    # R1
        pivots[i, 6] = c + rng * 1.1 / 6     # R2
        pivots[i, 7] = c + rng * 1.1 / 4     # R3
        pivots[i, 8] = c + rng * 1.1 / 3     # R4
        pivots[i, 9] = c + rng * 1.1 / 2     # R5 (extreme)
    
    return pivots

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
    """Choppiness Index - lower = trending, higher = choppy"""
    n = len(high)
    chop = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        tr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            if j > 0:
                tr = max(high[j] - low[j], abs(high[j] - close[j-1]))
            else:
                tr = high[j] - low[j]
            tr_sum += tr
        
        if tr_sum > 0:
            hh = np.max(high[i - period + 1:i + 1])
            ll = np.min(low[i - period + 1:i + 1])
            range_hl = hh - ll
            
            if range_hl > 0:
                chop[i] = 100 * np.log10(tr_sum / range_hl) / np.log10(period)
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE ===
    df_1w = get_htf_data(prices, '1w')
    
    # 1w SMA200 for long-term trend
    sma_1w = pd.Series(df_1w['close'].values).rolling(window=200, min_periods=200).mean().values
    sma_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_1w)
    
    # 1w SMA50 for trend direction (faster)
    sma50_1w = pd.Series(df_1w['close'].values).rolling(window=50, min_periods=50).mean().values
    sma50_1w_aligned = align_htf_to_ltf(prices, df_1w, sma50_1w)
    
    # Local indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    
    # Volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Camarilla levels (using 24-period = 12 days for 12h)
    # We'll calculate S3, S4, R3, R4
    s3_levels = np.full(n, np.nan)
    s4_levels = np.full(n, np.nan)
    r3_levels = np.full(n, np.nan)
    r4_levels = np.full(n, np.nan)
    
    for i in range(24, n):
        h = high[i]
        l = low[i]
        c = close[i]
        rng = h - l
        
        s3_levels[i] = c - rng * 1.1 / 4   # S3
        s4_levels[i] = c - rng * 1.1 / 3   # S4
        r3_levels[i] = c + rng * 1.1 / 4   # R3
        r4_levels[i] = c + rng * 1.1 / 3   # R4
    
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
    
    warmup = 250  # Need enough for 1w SMA200 + Camarilla(24)
    
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
        
        if np.isnan(chop[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === TREND DIRECTION (1w SMA200) ===
        price_above_1w_sma = close[i] > sma_1w_aligned[i]
        sma_1w_rising = sma50_1w_aligned[i] > sma_1w_aligned[i] if not np.isnan(sma50_1w_aligned[i]) else price_above_1w_sma
        
        is_bull_trend = price_above_1w_sma and sma_1w_rising
        is_bear_trend = not price_above_1w_sma and not sma_1w_rising
        
        # === REGIME (Choppiness Index) ===
        # Skip if too choppy (CHOP > 61.8) when entering new position
        is_choppy = chop[i] > 61.8
        is_trending = chop[i] < 50.0
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.3
        
        # === CAMARILLA LEVEL TOUCH ===
        # Check if price touches S3/S4 or R3/R4
        prev_close = close[i - 1] if i > 0 else close[i]
        
        # Long signal: price drops to S3/S4 zone
        touch_s3 = (low[i] <= s3_levels[i]) and (prev_close > s3_levels[i])
        touch_s4 = (low[i] <= s4_levels[i]) and (prev_close > s4_levels[i])
        
        # Short signal: price rises to R3/R4 zone
        touch_r3 = (high[i] >= r3_levels[i]) and (prev_close < r3_levels[i])
        touch_r4 = (high[i] >= r4_levels[i]) and (prev_close < r4_levels[i])
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG: Touch S3/S4 in bull trend with volume ===
            if (touch_s3 or touch_s4) and is_bull_trend:
                if vol_spike or is_trending:
                    desired_signal = SIZE
            
            # === SHORT: Touch R3/R4 in bear trend with volume ===
            if (touch_r3 or touch_r4) and is_bear_trend:
                if vol_spike or is_trending:
                    desired_signal = -SIZE
        
        # === STOPLOSS CHECK (2.5 ATR) ===
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
        
        # === TIME-BASED EXIT (hold at least 4 bars = 2 days) ===
        bars_held = i - entry_bar
        
        if in_position and bars_held >= 4:
            # Exit if price reverts to Camarilla middle
            midpoint = (s3_levels[i] + r3_levels[i]) / 2 if not np.isnan(s3_levels[i]) else close[i]
            
            if position_side > 0 and close[i] > midpoint + 0.5 * atr_14[i]:
                # Target hit: +2R, reduce size
                desired_signal = SIZE / 2
            
            if position_side < 0 and close[i] < midpoint - 0.5 * atr_14[i]:
                # Target hit: +2R, reduce size
                desired_signal = -SIZE / 2
        
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