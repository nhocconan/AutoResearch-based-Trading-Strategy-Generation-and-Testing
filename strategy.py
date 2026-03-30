#!/usr/bin/env python3
"""
Experiment #021: 12h Camarilla S4/R4 + 1d Donchian + 1d EMA50 Trend

HYPOTHESIS: Buy S4 touches (deep support) when 1d trend is bullish AND 1d
Donchian(20) confirms uptrend. Short R4 touches when 1d trend is bearish AND
Donchian confirms downtrend. This triple confirmation (Camarilla + EMA + Donchian)
should capture reversals at key institutional levels with minimal false signals.

WHY 12h: ~3x fewer bars than 4h = 3x fewer potential trades = less fee drag.
12h Camarilla captures multi-day swing reversals.

WHY IT WORKS IN BULL AND BEAR: S4 in uptrend = buy the dip (bull).
R4 in downtrend = short the rally (bear). Symmetrical edge.

TARGET: 75-150 total trades over 4 years. HARD MAX: 200.
Signal size: 0.25.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_camarilla_s4r4_donchian_1d_v1"
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
    """Donchian Channel - upper = highest high, lower = lowest low"""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

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
    
    # 1d Donchian(20) for trend confirmation
    donchian_1d_upper = pd.Series(df_1d['high'].values).rolling(window=20, min_periods=20).max().values
    donchian_1d_lower = pd.Series(df_1d['low'].values).rolling(window=20, min_periods=20).min().values
    donchian_1d_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_1d_upper)
    donchian_1d_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_1d_lower)
    
    # === Local 12h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Volume ratio (20-period MA)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Choppiness Index for regime filtering
    # CI = 100 * log10(sum(ATR14) / (max(H-L) over period)) / log10(period)
    # < 38.2 = trending, > 61.8 = ranging
    chop_period = 14
    chop_values = np.full(n, np.nan)
    for i in range(chop_period, n):
        tr_sum = 0.0
        for j in range(chop_period):
            tr = max(high[i-j] - low[i-j], abs(high[i-j] - close[i-j-1]) if i-j > 0 else high[i-j] - low[i-j])
            tr_sum += tr
        hL = max(high[i-chop_period+1:i+1]) - min(low[i-chop_period+1:i+1])
        if hL > 0 and tr_sum > 0:
            chop_values[i] = 100 * np.log10(tr_sum / hL) / np.log10(chop_period)
    
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
    min_hold_bars = 2  # Minimum 2 bars (1 day) before considering exit
    
    warmup = 100  # Need enough for all indicators
    
    for i in range(warmup, n):
        # Skip if ATR not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # Skip if HTF indicators not aligned
        if np.isnan(ema_1d_aligned[i]) or np.isnan(donchian_1d_upper_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === TREND CONFIRMATION (1d EMA50 + Donchian) ===
        price_above_1d_ema = close[i] > ema_1d_aligned[i]
        price_below_1d_ema = close[i] < ema_1d_aligned[i]
        
        # 1d Donchian: confirm trend by checking if price is in upper/lower third
        donchian_mid = (donchian_1d_upper_aligned[i] + donchian_1d_lower_aligned[i]) / 2
        donchian_range = donchian_1d_upper_aligned[i] - donchian_1d_lower_aligned[i]
        price_near_donchian_top = close[i] > donchian_mid + donchian_range * 0.25  # upper quartile
        price_near_donchian_bot = close[i] < donchian_mid - donchian_range * 0.25  # lower quartile
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] >= 2.0  # Stronger volume filter
        
        # === CHOPPINESS FILTER ===
        chop = chop_values[i]
        is_trending = not np.isnan(chop) and chop < 55.0
        
        # === CAMARILLA LEVELS from previous CLOSED bar (no look-ahead) ===
        prev_high = high[i - 1]
        prev_low = low[i - 1]
        prev_close = close[i - 1]
        prev_range = prev_high - prev_low
        
        # Classic Camarilla levels (factor 1.1/12 = 0.09167, 2.2/12 = 0.18333)
        r4 = prev_close + prev_range * 0.18333
        s4 = prev_close - prev_range * 0.18333
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG: Price touches S4 with TRIPLE confirmation ===
            # 1. Price above 1d EMA50 (trend up)
            # 2. Price in upper quartile of 1d Donchian (confirmed uptrend)
            # 3. Volume spike (> 2x average)
            # 4. S4 touch (deep support pullback)
            if price_above_1d_ema and price_near_donchian_top and vol_spike:
                if low[i] <= s4:
                    desired_signal = SIZE
            
            # === SHORT: Price touches R4 with TRIPLE confirmation ===
            # 1. Price below 1d EMA50 (trend down)
            # 2. Price in lower quartile of 1d Donchian (confirmed downtrend)
            # 3. Volume spike
            # 4. R4 touch (deep resistance rally)
            if price_below_1d_ema and price_near_donchian_bot and vol_spike:
                if high[i] >= r4:
                    desired_signal = -SIZE
        
        # === STOPLOSS (2.0 ATR trailing) ===
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.0 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.0 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                desired_signal = 0.0
        
        # === HOLD PERIOD + TAKE PROFIT ===
        bars_held = i - entry_bar
        
        if in_position and bars_held >= min_hold_bars:
            # Exit when price reverts to previous close ( Camarilla mid)
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
                    stop_price = entry_price - 2.0 * entry_atr
                else:
                    stop_price = entry_price + 2.0 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                stop_price = 0.0
        
        signals[i] = desired_signal
    
    return signals