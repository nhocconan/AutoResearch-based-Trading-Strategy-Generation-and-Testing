#!/usr/bin/env python3
"""
Experiment #008: 12h Camarilla S4/R4 + Volume Spike + Weekly Trend + Choppiness

HYPOTHESIS: Camarilla S4/R4 levels are where institutional stop hunts occur.
Combined with:
- 1w EMA200 trend filter (removes counter-trend trades)
- Choppiness < 50 (trending market only = higher win rate)
- Volume spike > 2.5x (smart money confirmation)
- ATR stoploss (2.5x for 12h = ~2 days of volatility)

WHY 12h: 2x slower than 4h = fewer trades = less fee drag.
Weekly trend filter ensures we only trade with macro direction.

TARGET: 75-150 total over 4 years = 19-37/year. HARD MAX: 200.
Signal size: 0.25-0.30.

LEARNED FROM FAILURES:
- #005 had 275 trades (too many) → add choppiness + weekly trend
- #016 had 1155 trades (massive overtrading) → use discrete levels only
- Donchian-only strategies fail → use Camarilla for better risk/reward
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_camarilla_s4_vol_chop_1w_v1"
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
    Choppiness Index - measures market choppiness vs trending
    CHOP > 61.8 = choppy (mean reversion)
    CHOP < 38.2 = trending (trend following)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    chop = np.full(n, np.nan)
    
    for i in range(period, n):
        sum_tr = 0.0
        for j in range(period):
            tr = max(high[i - j] - low[i - j], 
                     abs(high[i - j] - close[i - j - 1] if i - j - 1 >= 0 else high[i - j]),
                     abs(low[i - j] - close[i - j - 1] if i - j - 1 >= 0 else low[i - j]))
            sum_tr += tr
        
        highest_high = max(high[i - period + 1:i + 1])
        lowest_low = min(low[i - period + 1:i + 1])
        range_val = highest_high - lowest_low
        
        if range_val > 0:
            chop[i] = 100 * (np.log(sum_tr) / np.log(range_val * period)) if range_val > 0 else 50
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    
    # 1w EMA200 for macro trend
    ema_1w = pd.Series(df_1w['close'].values).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # 1d EMA50 for medium trend
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # === Local 12h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    
    # Volume ratio (20-bar to avoid false spikes)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Signals
    signals = np.zeros(n)
    SIZE = 0.25
    CHOP_THRESHOLD = 50.0  # Only trend when CHOP < 50
    VOL_THRESHOLD = 2.5    # Strong volume confirmation
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_bar = 0
    
    warmup = 500  # Need enough for 1w EMA200 alignment
    
    for i in range(warmup, n):
        # Skip if ATR not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # Skip if weekly EMA not aligned
        if np.isnan(ema_1w_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # Skip if daily EMA not aligned
        if np.isnan(ema_1d_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === TREND FILTERS ===
        # 1w EMA200 for macro trend
        price_above_1w_ema = close[i] > ema_1w_aligned[i]
        # 1d EMA50 for medium trend
        price_above_1d_ema = close[i] > ema_1d_aligned[i]
        
        # === CHOPPINESS FILTER (only trade in trending markets) ===
        chop = chop_14[i]
        is_trending = not np.isnan(chop) and chop < CHOP_THRESHOLD
        
        # Volume confirmation (stronger threshold to reduce false entries)
        vol_spike = vol_ratio[i] > VOL_THRESHOLD
        
        # === CAMARILLA LEVELS from previous CLOSED bar (no look-ahead) ===
        prev_high = high[i - 1]
        prev_low = low[i - 1]
        prev_close = close[i - 1]
        prev_range = prev_high - prev_low
        
        # Classic Camarilla levels
        r3 = prev_close + prev_range * 0.09167
        r4 = prev_close + prev_range * 0.18333
        s3 = prev_close - prev_range * 0.09167
        s4 = prev_close - prev_range * 0.18333
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG: Price touches S4 with ALL confirmations ===
            # S4 is the outer level - more extreme = stronger signal
            if price_above_1w_ema and price_above_1d_ema and is_trending and vol_spike:
                if low[i] <= s4:
                    desired_signal = SIZE
            
            # === SHORT: Price touches R4 with ALL confirmations ===
            if not price_above_1w_ema and not price_above_1d_ema and is_trending and vol_spike:
                if high[i] >= r4:
                    desired_signal = -SIZE
        
        # === STOPLOSS (2.5x ATR trailing = wider for 12h) ===
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                desired_signal = 0.0
        
        # === HOLD PERIOD (minimum 3 bars = 1.5 days to avoid churn) ===
        bars_held = i - entry_bar
        
        if in_position and bars_held >= 3:
            # Take profit if price reverts to Camarilla mid (prev close)
            if position_side > 0 and close[i] >= prev_close:
                desired_signal = 0.0
            if position_side < 0 and close[i] <= prev_close:
                desired_signal = 0.0
        
        # === UPDATE POSITION ===
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
                stop_price = 0.0
        
        signals[i] = desired_signal
    
    return signals