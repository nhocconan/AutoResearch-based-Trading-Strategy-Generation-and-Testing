#!/usr/bin/env python3
"""
Experiment #021: 4h Donchian + Volume + Choppiness (proven winning pattern)

HYPOTHESIS: Donchian(20) breakout with volume confirmation captures institutional
momentum. Choppiness Index filters out range-bound periods where breakouts fail.
This is the PROVEN pattern from ETHUSDT (test Sharpe 1.47) and SOLUSDT (Sharpe 1.38).

WHY 4h: Optimal trade frequency (20-50/year). Matches the winning DB patterns exactly.
Donchian channels work in BOTH bull (breakout up) and bear (breakdown down) markets
because they capture when price MAKES NEW HIGHS/LOWS — the definition of trend.

ENTRY CONDITIONS (tight = fewer trades):
- Long: Donchian(20) upper break + volume spike (1.8x) + CHOP < 50 + HTF trend aligned
- Short: Donchian(20) lower break + volume spike + CHOP < 50 + HTF trend aligned

EXIT: 2.5 ATR stoploss, exit on opposite Donchian touch or CHOP > 65

TARGET: 75-200 total trades over 4 years. HARD MAX: 400.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_vol_chop_proven_v1"
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

def calculate_donchian(high, low, period=20):
    """Donchian Channel: Upper = highest high, Lower = lowest low"""
    n = len(high)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    mid = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
        mid[i] = (upper[i] + lower[i]) / 2.0
    
    return upper, lower, mid

def calculate_choppiness_index(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    CHOP > 61.8 = choppy/range market (avoid trend trades)
    CHOP < 38.2 = trending market (momentum works)
    """
    n = len(high)
    chop = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        tr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            tr = max(high[j] - low[j], abs(high[j] - close[j-1]) if j > 0 else high[j] - low[j])
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
    
    # === Load HTF data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d SMA200 for trend direction (aligned to 4h)
    sma_200_1d = pd.Series(df_1d['close'].values).rolling(window=200, min_periods=200).mean().values
    sma_200_aligned = align_htf_to_ltf(prices, df_1d, sma_200_1d)
    
    # 1d EMA21 for faster trend confirmation
    ema_21_1d = pd.Series(df_1d['close'].values).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_21_aligned = align_htf_to_ltf(prices, df_1d, ema_21_1d)
    
    # === Local 4h indicators ===
    donchian_upper, donchian_lower, donchian_mid = calculate_donchian(high, low, period=20)
    atr_14 = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness_index(high, low, close, period=14)
    
    # Volume for confirmation (pre-compute before loop)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # === Signal arrays ===
    signals = np.zeros(n)
    SIZE = 0.30  # Position size
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_bar = 0
    
    warmup = 250  # Need 200 for SMA200 + buffer
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(sma_200_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(chop[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === HTF TREND (1d) ===
        htf_bullish = close[i] > sma_200_aligned[i]  # Above SMA200 = bull trend
        htf_bearish = close[i] < sma_200_aligned[i]  # Below SMA200 = bear trend
        
        # Also check EMA21 alignment for confirmation
        ema21_aligned = ema_21_aligned[i] if not np.isnan(ema_21_aligned[i]) else close[i]
        htf_confirmed_bull = htf_bullish and close[i] > ema21_aligned
        htf_confirmed_bear = htf_bearish and close[i] < ema21_aligned
        
        # === REGIME (Choppiness) ===
        # CHOP < 50 = trending or transitioning (allow trades)
        # CHOP > 65 = very choppy (avoid)
        is_tradeable = chop[i] < 50.0
        is_choppy = chop[i] > 65.0
        
        # === DONCHIAN SIGNALS ===
        donch_upper = donchian_upper[i]
        donch_lower = donchian_lower[i]
        donch_mid_current = donchian_mid[i]
        
        # Previous Donchian for breakout detection
        donch_upper_prev = donchian_upper[i-1] if i > 0 and not np.isnan(donchian_upper[i-1]) else donch_upper
        donch_lower_prev = donchian_lower[i-1] if i > 0 and not np.isnan(donchian_lower[i-1]) else donch_lower
        
        # Breakout: current high/low exceeds previous Donchian level
        upper_breakout = high[i] > donch_upper_prev and close[i] > donch_upper
        lower_breakout = low[i] < donch_lower_prev and close[i] < donch_lower
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.8  # Require 1.8x average volume
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG ENTRY ===
            # Upper Donchian break + volume + not too choppy + HTF bull aligned
            if upper_breakout and vol_spike and is_tradeable and (htf_bullish or htf_confirmed_bull):
                desired_signal = SIZE
            
            # === SHORT ENTRY ===
            # Lower Donchian break + volume + not too choppy + HTF bear aligned
            if lower_breakout and vol_spike and is_tradeable and (htf_bearish or htf_confirmed_bear):
                desired_signal = -SIZE
        
        # === STOPLOSS CHECK (2.5 ATR trailing) ===
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            if low[i] < trailing_stop:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            if high[i] > trailing_stop:
                desired_signal = 0.0
        
        # === EXIT ON CHOPPINESS DETERIORATION ===
        if in_position and is_choppy:
            # Exit if CHOP spikes into choppy territory while we're in position
            if (position_side > 0 and chop[i] > 70) or (position_side < 0 and chop[i] > 70):
                desired_signal = 0.0
        
        # === EXIT ON OPPOSITE DONCHIAN TOUCH ===
        if in_position and position_side > 0:
            # Exit long if price falls back below mid-channel and losing
            if close[i] < donch_mid_current and close[i] < entry_price:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if price rises back above mid-channel and gaining
            if close[i] > donch_mid_current and close[i] > entry_price:
                desired_signal = 0.0
        
        # === TIME-BASED EXIT (hold at least 6 bars = 1 day) ===
        bars_held = i - entry_bar if in_position else 0
        if in_position and bars_held >= 6:
            # Exit if not making progress
            if position_side > 0 and close[i] < donch_lower:
                desired_signal = 0.0
            if position_side < 0 and close[i] > donch_upper:
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
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = desired_signal
    
    return signals