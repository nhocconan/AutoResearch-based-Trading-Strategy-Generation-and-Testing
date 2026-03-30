#!/usr/bin/env python3
"""
Experiment #008: 12h Donchian Breakout + 1w Trend + Volume

HYPOTHESIS: Price structure breakout on 12h captures multi-day momentum shifts.
By combining:
1. 1w EMA(21) for structural trend (slow, reliable)
2. 12h Donchian(20) for breakout signals
3. Volume confirmation for institutional validity
4. 2.5x ATR trailing stop for risk management

This captures the core pattern from DB winners: Donchian + volume + regime filter.

WHY 12h: ~3x fewer trades than 4h = less fee drag, more meaningful signals.
Target: 75-150 total over 4 years (19-37/year).

Signal size: 0.25.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_vol_1w_v1"
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
    
    # === Load HTF data ONCE before loop ===
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly EMA(21) for structural trend direction
    ema_1w = pd.Series(df_1w['close'].values).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # === Local 12h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Donchian Channel (20 bars = ~10 days)
    donch_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    donch_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Volume ratio (confirm institutional activity)
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
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_bar = 0
    
    warmup = 150  # Need enough for all indicators
    
    for i in range(warmup, n):
        # Skip if ATR not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # Skip if 1w EMA not aligned
        if np.isnan(ema_1w_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        desired_signal = 0.0
        
        # === 1w TREND DIRECTION (bullish if rising, bearish if falling) ===
        bullish_trend = ema_1w_aligned[i] > ema_1w_aligned[i - 5] if i >= 5 else False
        bearish_trend = ema_1w_aligned[i] < ema_1w_aligned[i - 5] if i >= 5 else False
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === ENTRY LOGIC ===
        if not in_position:
            # === LONG: Price breaks above 20-bar high with volume + uptrend ===
            # Donchian high crossed - price making new 20-bar high
            price_new_high = close[i] > donch_high_20[i] if not np.isnan(donch_high_20[i]) else False
            if bullish_trend and vol_spike and price_new_high:
                desired_signal = SIZE
            
            # === SHORT: Price breaks below 20-bar low with volume + downtrend ===
            price_new_low = close[i] < donch_low_20[i] if not np.isnan(donch_low_20[i]) else False
            if bearish_trend and vol_spike and price_new_low:
                desired_signal = -SIZE
        
        # === TRAILING STOP (2.5x ATR) ===
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            if low[i] < trailing_stop:
                desired_signal = 0.0
            else:
                desired_signal = SIZE
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            if high[i] > trailing_stop:
                desired_signal = 0.0
            else:
                desired_signal = -SIZE
        
        # === FORCE DIRECTION IF IN POSITION ===
        if in_position:
            desired_signal = SIZE * position_side
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                entry_bar = i
        else:
            if in_position:
                in_position = False
                position_side = 0
        
        signals[i] = desired_signal
    
    return signals