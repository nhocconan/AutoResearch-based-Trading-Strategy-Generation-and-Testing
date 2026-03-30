#!/usr/bin/env python3
"""
Experiment #028: 12h Donchian Breakout + 1d EMA21 Trend + Choppiness Regime

HYPOTHESIS: 12h Donchian(20) breakout is a proven price channel structure.
Combined with 1d EMA21 for faster trend direction (vs slow SMA200 that caused 0 trades),
and Choppiness Index to filter range-bound markets.

WHY 12h: Natural trade limiter (50-100/year) vs 6h which overtraded (459 trades).
WHY 1d EMA21: Faster than SMA200 (adjusts to trend faster = more signals).
EMA21 1d = ~3 week trend, fast enough for signals but stable.

TARGET: 50-150 total over 4 years (12-37/year). HARD MAX: 200.
Works in BOTH bull (long breakouts) and bear (short breakdowns).
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_1d_ema21_chop_v1"
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
    
    # === Load 1d HTF data ONCE ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA21 for trend direction (faster than SMA200)
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Local 12h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    
    # Donchian channels (20 periods on 12h = 10 days)
    donchian_period = 20
    donchian_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # Volume average
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
    
    warmup = 80  # Need enough for Donchian(20) + indicators
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(ema_1d_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(chop[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === TREND DIRECTION (1d EMA21) ===
        price_above_1d_ema = close[i] > ema_1d_aligned[i]
        
        # === REGIME (Choppiness Index) ===
        # Skip if too choppy (CHOP > 61.8) — only take new trades in trending markets
        is_choppy = chop[i] > 61.8
        
        if is_choppy and not in_position:
            signals[i] = 0.0
            continue
        
        # === DONCHIAN BREAKOUT SIGNALS ===
        current_high = high[i]
        current_low = low[i]
        
        # Previous bar's Donchian values (use i-1 to avoid look-ahead)
        prev_donchian_high = donchian_high[i - 1] if i > 0 else 0
        prev_donchian_low = donchian_low[i - 1] if i > 0 else 0
        
        # Volume confirmation (optional but helps)
        vol_spike = vol_ratio[i] > 1.5
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG: Breakout above Donchian high + uptrend ===
            # Price closes above previous 20-bar high with volume
            if current_high > prev_donchian_high and price_above_1d_ema:
                # Optional volume filter — only require volume if choppy
                if vol_spike or chop[i] < 61.8:
                    desired_signal = SIZE
            
            # === SHORT: Breakdown below Donchian low + downtrend ===
            if current_low < prev_donchian_low and not price_above_1d_ema:
                if vol_spike or chop[i] < 61.8:
                    desired_signal = -SIZE
        
        # === STOPLOSS CHECK (2.5 ATR trailing) ===
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
            # Exit if price crosses Donchian mid-point
            donchian_mid = (donchian_high[i] + donchian_low[i]) / 2
            if position_side > 0 and close[i] < donchian_mid:
                desired_signal = 0.0
            if position_side < 0 and close[i] > donchian_mid:
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