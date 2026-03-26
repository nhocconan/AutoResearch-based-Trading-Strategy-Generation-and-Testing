#!/usr/bin/env python3
"""
Experiment #021: 12h Donchian Breakout + 1d Trend + Volume Confirmation

HYPOTHESIS: Donchian(20) breakout is a proven price structure signal that captures
trend changes. Combined with 1d SMA50 for trend direction and volume confirmation,
this catches institutional moves. Donchian is particularly effective on higher TFs
(12h+) because it filters noise while remaining responsive.

WHY 12h: Slower than 4h (reduces fee drag, more stable signals), more opportunities
than 1d. Works in both bull (2021) and bear (2022) because we filter by 1d trend.

KEY DIFFERENCE FROM FAILED STRATEGIES: 
- Simple Donchian breakout (not complex multi-indicator stacking)
- 1d SMA50 for trend (not RSI extremes that rarely trigger)
- Volume spike confirmation (filters false breakouts)
- 12h Donchian (20) = 10 days of data, good balance

TARGET: 50-150 total trades over 4 years = 12-37/year. HARD MAX: 200.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_1d_sma50_vol_v1"
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

def calculate_donchian(high, low, period=20):
    """
    Donchian Channel
    Upper = highest high over period
    Lower = lowest low over period
    Middle = (Upper + Lower) / 2
    """
    n = len(high)
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    middle = (upper + lower) / 2
    return upper, middle, lower

def calculate_choppiness_index(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    CHOP > 61.8 = choppy/range market
    CHOP < 38.2 = trending market
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
    
    # === Load HTF data ONCE ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d SMA50 for trend direction
    sma_1d = pd.Series(df_1d['close'].values).rolling(window=50, min_periods=50).mean().values
    sma_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_1d)
    
    # Local 12h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    donchian_upper, donchian_mid, donchian_lower = calculate_donchian(high, low, period=20)
    chop = calculate_choppiness_index(high, low, close, period=14)
    
    # Volume for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    signals = np.zeros(n)
    SIZE = 0.25  # Conservative sizing for 12h
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_bar = 0
    
    warmup = 100  # Need 50 for SMA50 + buffer
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
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
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === TREND DIRECTION (1d SMA50) ===
        price_above_1d_sma = close[i] > sma_1d_aligned[i]
        
        # === REGIME (Choppiness Index) - less restrictive ===
        # CHOP < 50 = trending (allow more trades)
        is_trending = chop[i] < 50.0
        is_choppy = chop[i] > 61.8
        
        # === DONCHIAN SIGNALS ===
        upper = donchian_upper[i]
        lower = donchian_lower[i]
        mid = donchian_mid[i]
        
        # Previous bar's Donchian for crossover detection
        upper_prev = donchian_upper[i-1] if i > 1 else upper
        lower_prev = donchian_lower[i-1] if i > 1 else lower
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.3
        
        # === ATR for stoploss ===
        atr_local = atr_14[i]
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === BULL TREND: Long on upper Donchian breakout ===
            if price_above_1d_sma:
                # Price breaks above upper Donchian with volume
                if close[i] > upper and vol_spike:
                    desired_signal = SIZE
                # Alternative: price near upper Donchian in strong trend
                elif close[i] > upper * 0.98 and is_trending and vol_spike:
                    desired_signal = SIZE
            
            # === BEAR TREND: Short on lower Donchian breakdown ===
            if not price_above_1d_sma:
                # Price breaks below lower Donchian with volume
                if close[i] < lower and vol_spike:
                    desired_signal = -SIZE
                # Alternative: price near lower Donchian in downtrend
                elif close[i] < lower * 1.02 and is_trending and vol_spike:
                    desired_signal = -SIZE
            
            # === CHOPPY MARKET: Mean reversion at Donchian extremes ===
            if is_choppy:
                # Long at lower Donchian in choppy (buying the dip)
                if close[i] < lower * 1.02:
                    desired_signal = SIZE * 0.5  # Half size in choppy
                
                # Short at upper Donchian in choppy (selling the rally)
                if close[i] > upper * 0.98:
                    desired_signal = -SIZE * 0.5
        
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
        
        # === TIME-BASED EXIT (hold at least 3 bars = 1.5 days) ===
        bars_held = i - entry_bar if in_position else 0
        min_hold_bars = 3
        
        if in_position and bars_held >= min_hold_bars:
            # Exit if trend reverses
            if position_side > 0 and not price_above_1d_sma:
                desired_signal = 0.0
            if position_side < 0 and price_above_1d_sma:
                desired_signal = 0.0
            # Exit if price crosses mid-Donchian
            if position_side > 0 and close[i] < mid:
                desired_signal = 0.0
            if position_side < 0 and close[i] > mid:
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