#!/usr/bin/env python3
"""
Experiment #054: 1h Session Filter + 4h Donchian Breakout + 1d Volume Spike

HYPOTHESIS: Trade 1h timeframe with strict session filter (08-20 UTC) to avoid Asian session noise.
Use 4h Donchian(20) breakout for structure and 1d volume spike (>2x 20-day average) for confirmation.
Only trade in direction of 4h trend (price > 4h EMA50). Target 15-37 trades/year to minimize fee drag.
Position size fixed at 0.20 (20%) with ATR-based stoploss (2.5x) and time-based exit (max 48h hold).
Designed to work in both bull (breakouts) and bear (short breakdowns) markets.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_session_donchian_volume_v1"
timeframe = "1h"
leverage = 1.0

def calculate_ema(values, period):
    """Exponential Moving Average"""
    return pd.Series(values).ewm(span=period, min_periods=period, adjust=False).mean().values

def calculate_atr(high, low, close, period=14):
    """Average True Range for stoploss calculation."""
    n = len(close)
    if n < 2:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    open_time = prices["open_time"].values  # Already datetime64[ms]
    n = len(close)
    
    # Pre-compute session hours (08-20 UTC) - avoid .astype('datetime64[ms]') crash
    hours = pd.DatetimeIndex(open_time).hour
    
    # === HTF: 4h EMA50 for trend filter (Call ONCE before loop) ===
    df_4h = get_htf_data(prices, '4h')
    ema_4h_50 = calculate_ema(df_4h['close'].values, 50)
    ema_4h_50_aligned = align_htf_to_ltf(prices, df_4h, ema_4h_50)
    
    # === HTF: 1d volume MA for spike detection (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    vol_1d = df_1d['volume'].values
    vol_ma_1d_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d_20)
    
    # === 1h Indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    dc_upper_20 = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    dc_lower_20 = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20  # Fixed position size (20% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_bar = -1
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    warmup = 50  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Session Filter: 08-20 UTC only ---
        if not (8 <= hours[i] <= 20):
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(atr_14[i]) or np.isnan(dc_upper_20[i]) or np.isnan(dc_lower_20[i]) or 
            np.isnan(ema_4h_50_aligned[i]) or np.isnan(vol_ma_1d_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- 4h Trend Filter ---
        trend_bullish = close[i] > ema_4h_50_aligned[i]
        trend_bearish = close[i] < ema_4h_50_aligned[i]
        
        # --- 1h Donchian Breakout ---
        bullish_breakout = close[i] > dc_upper_20[i]
        bearish_breakout = close[i] < dc_lower_20[i]
        
        # --- 1d Volume Spike (>2x 20-day average) ---
        vol_spike = volume[i] > vol_ma_1d_20_aligned[i] * 2.0 if vol_ma_1d_20_aligned[i] > 1e-10 else False
        
        # --- Position Management (Exit Logic) ---
        stop_hit = False
        
        if in_position:
            # Time-based exit: max 48 hours (48 bars on 1h)
            max_hold_bars = 48
            time_exit = (i - entry_bar) >= max_hold_bars
            
            # ATR-based stoploss
            if position_side > 0:
                stop_level = entry_price - 2.5 * atr_14[i]
                if low[i] < stop_level:
                    stop_hit = True
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14[i]
                if high[i] > stop_level:
                    stop_hit = True
            
            # Exit conditions: trend reversal or opposite Donchian touch
            min_hold = (i - entry_bar) >= 4  # Minimum 4 bars hold (~4h)
            if min_hold and not time_exit:
                if position_side > 0:
                    # Exit long: trend turns bearish OR price touches lower Donchian
                    if trend_bearish or close[i] <= dc_lower_20[i]:
                        stop_hit = True
                else:  # position_side < 0
                    # Exit short: trend turns bullish OR price touches upper Donchian
                    if trend_bullish or close[i] >= dc_upper_20[i]:
                        stop_hit = True
            
            if stop_hit or time_exit:
                signals[i] = 0.0
                in_position = False
                position_side = 0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
            else:
                signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Require volume spike to avoid low-volume false breakouts
        if vol_spike:
            # Long conditions: 
            # Breakout above upper Donchian with bullish 4h EMA trend
            if bullish_breakout and trend_bullish:
                in_position = True
                position_side = 1
                entry_bar = i
                entry_price = close[i]
                highest_since_entry = high[i]
                signals[i] = SIZE
            # Short conditions:
            # Breakout below lower Donchian with bearish 4h EMA trend
            elif bearish_breakout and trend_bearish:
                in_position = True
                position_side = -1
                entry_bar = i
                entry_price = close[i]
                lowest_since_entry = low[i]
                signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals