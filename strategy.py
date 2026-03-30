#!/usr/bin/env python3
"""
Experiment #021: 4h Camarilla S4/R4 + Volume Spike + Choppiness Regime

HYPOTHESIS: Camarilla S4 and R4 are deep support/resistance levels where institutions
accumulate/distribute. By combining with:
1. 1d EMA50 for trend direction (bull vs bear)
2. Volume spike > 2x average (institutional confirmation)
3. Choppiness Index < 61.8 (trending environment only)

This catches major reversals at key levels while avoiding ranging chop.

WHY IT WORKS IN BOTH BULL AND BEAR MARKETS:
- Bull: Buy S4 touches when price > 1d EMA50 (accumulation zones)
- Bear: Short R4 touches when price < 1d EMA50 (distribution zones)
- Symmetrical approach adapts to any market regime

TARGET: 75-150 total trades over 4 years. HARD MAX: 200.
Signal size: 0.25.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_camarilla_s4r4_vol_chop_1d_v1"
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

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    CHOP > 61.8 = ranging (use mean reversion)
    CHOP < 38.2 = trending (use trend following)
    Values outside 38.2-61.8 = transition zone
    
    Formula: 100 * LOG10(SUM(ATR(1),period) / (HHV(period) - LLV(period))) / LOG10(period)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    for i in range(period, n):
        atr_sum = 0.0
        for j in range(period):
            tr = max(high[i - j] - low[i - j], 
                     abs(high[i - j] - close[i - j - 1]) if i - j - 1 >= 0 else high[i - j] - low[i - j],
                     abs(low[i - j] - close[i - j - 1]) if i - j - 1 >= 0 else high[i - j] - low[i - j])
            atr_sum += tr
        
        hh = max(high[i - period + 1:i + 1])
        ll = min(low[i - period + 1:i + 1])
        
        if hh > ll and atr_sum > 0:
            chop[i] = 100 * np.log10(atr_sum / (hh - ll)) / np.log10(period)
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA50 for trend direction
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # === Local 4h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    
    # Volume ratio (20-bar average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Signals
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
    
    warmup = 100  # Need enough for indicators + alignment buffer
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(ema_1d_aligned[i]) or np.isnan(chop_14[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === REGIME FILTER: Choppiness < 61.8 (trending environment) ===
        is_trending = chop_14[i] < 61.8
        
        # === TREND DIRECTION (1d EMA50) ===
        price_above_1d_ema = close[i] > ema_1d_aligned[i]
        
        # === VOLUME CONFIRMATION (2x average = strong institutional interest) ===
        vol_spike = vol_ratio[i] > 2.0
        
        # === CAMARILLA LEVELS from previous CLOSED bar (no look-ahead) ===
        prev_high = high[i - 1]
        prev_low = low[i - 1]
        prev_close = close[i - 1]
        prev_range = prev_high - prev_low
        
        # Classic Camarilla S4 and R4 levels
        r4 = prev_close + prev_range * 0.18333
        s4 = prev_close - prev_range * 0.18333
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG: Price touches S4 with volume + trend + regime alignment ===
            # Only in trending markets (CHOP < 61.8)
            if is_trending and price_above_1d_ema and vol_spike:
                if low[i] <= s4:
                    desired_signal = SIZE
            
            # === SHORT: Price touches R4 with volume + trend + regime alignment ===
            if is_trending and not price_above_1d_ema and vol_spike:
                if high[i] >= r4:
                    desired_signal = -SIZE
        
        # === STOPLOSS (2.5 ATR trailing - slightly wider for 4h) ===
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
        
        # === MINIMUM HOLDING PERIOD (3 bars = 12h to avoid chop churn) ===
        bars_held = i - entry_bar
        
        if in_position and bars_held >= 3:
            # Take profit if price moves 3x ATR in our favor
            if position_side > 0:
                profit_target = entry_price + 3.0 * entry_atr
                if close[i] >= profit_target:
                    desired_signal = 0.0  # Exit with profit
            if position_side < 0:
                profit_target = entry_price - 3.0 * entry_atr
                if close[i] <= profit_target:
                    desired_signal = 0.0  # Exit with profit
        
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