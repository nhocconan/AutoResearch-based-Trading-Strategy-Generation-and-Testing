#!/usr/bin/env python3
"""
Experiment #005: 12h Donchian Breakout + Volume + 1d EMA21 Trend

HYPOTHESIS: Donchian(25) breakout on 12h captures multi-day trend changes.
Volume spike confirms institutional accumulation/distribution.
1d EMA21 alignment ensures we trade WITH the larger trend.

WHY 12h: Slower than 4h = fewer trades = less fee drag.
Donchian(25) on 12h = 25*12h = 300h = ~12.5 days channel.
This is tight enough to catch meaningful breakouts.

TARGET: 75-150 total trades over 4 years (19-37/year).
Signal size: 0.25.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_vol_ema21_1d_v1"
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
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA21 for trend direction
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # === Local 12h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Donchian channels - 25 bars (~12.5 days)
    donchian_period = 25
    rolling_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    rolling_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # Volume ratio
    vol_ma_period = 20
    vol_ma = pd.Series(volume).rolling(window=vol_ma_period, min_periods=vol_ma_period).mean().values
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
    
    warmup = max(100, donchian_period + vol_ma_period)
    
    for i in range(warmup, n):
        # Skip if ATR not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # Skip if HTF EMA not aligned
        if np.isnan(ema_1d_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === 1d TREND DIRECTION ===
        price_above_1d_ema = close[i] > ema_1d_aligned[i]
        price_below_1d_ema = close[i] < ema_1d_aligned[i]
        
        # Volume confirmation (1.5x average)
        vol_spike = vol_ratio[i] > 1.5
        
        # Donchian breakout levels (previous bar for no look-ahead)
        prev_donchian_high = rolling_high[i - 1]
        prev_donchian_low = rolling_low[i - 1]
        
        # Current bar breakout detection
        bullish_breakout = high[i] > prev_donchian_high
        bearish_breakout = low[i] < prev_donchian_low
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG: Bullish breakout + volume + 1d uptrend ===
            if bullish_breakout and vol_spike and price_above_1d_ema:
                desired_signal = SIZE
            
            # === SHORT: Bearish breakout + volume + 1d downtrend ===
            if bearish_breakout and vol_spike and price_below_1d_ema:
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
        
        # === TAKE PROFIT (3x ATR) ===
        if in_position and position_side > 0:
            profit_target = entry_price + 3.0 * entry_atr
            if high[i] >= profit_target:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            profit_target = entry_price - 3.0 * entry_atr
            if low[i] <= profit_target:
                desired_signal = 0.0
        
        # === ANTI-WHIPSAW: minimum 3 bars hold ===
        bars_held = i - entry_bar
        if in_position and bars_held < 3:
            # Keep position regardless of signals
            desired_signal = position_side * SIZE
        
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