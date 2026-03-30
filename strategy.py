#!/usr/bin/env python3
"""
Experiment #007: 6h Donchian(20) Breakout + 1d EMA21 Trend + Volume

HYPOTHESIS: Donchian channels define clear price structure boundaries.
Breakouts above/below these channels capture momentum shifts.
- LONG: price breaks above 6h Donchian high(20) + above 1d EMA21 + vol spike
- SHORT: price breaks below 6h Donchian low(20) + below 1d EMA21 + vol spike
- ATR trailing stop (2.5x) for risk management

WHY 6h: Slower than 4h = fewer trades = less fee drag.
WHY IT WORKS: Simple breakout mechanics, institutional attention to Donchian levels.
Works in both bull (buy breakouts) and bear (short breakdowns).

TARGET: 75-200 total trades over 4 years. HARD MAX: 300.
Signal size: 0.30.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_donchian_ema21_vol_v1"
timeframe = "6h"
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
    
    # === Local 6h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Donchian channels (20 periods)
    donchian_period = 20
    donchian_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # Volume ratio (20 period MA)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Signals
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    
    warmup = max(100, donchian_period)  # Need enough for Donchian + EMA alignment
    
    for i in range(warmup, n):
        # Skip if ATR not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # Skip if EMA not aligned
        if np.isnan(ema_1d_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # Skip if Donchian not ready
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === TREND DIRECTION (1d EMA21) ===
        price_above_1d_ema = close[i] > ema_1d_aligned[i]
        
        # Volume confirmation
        vol_spike = vol_ratio[i] > 1.5
        
        # Donchian levels from current bar (completed)
        d_high = donchian_high[i]
        d_low = donchian_low[i]
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG: Breakout above Donchian high + trend alignment + volume ===
            if price_above_1d_ema and vol_spike:
                if high[i] > d_high:
                    desired_signal = SIZE
            
            # === SHORT: Breakdown below Donchian low + trend alignment + volume ===
            if not price_above_1d_ema and vol_spike:
                if low[i] < d_low:
                    desired_signal = -SIZE
        
        # === STOPLOSS (2.5 ATR trailing) ===
        if in_position and position_side > 0:
            # Update trailing stop
            new_stop = high[i] - 2.5 * entry_atr
            if new_stop > entry_price - 2.5 * entry_atr:
                entry_price = entry_price  # Keep original entry
                # Only exit if price drops below stop
                if low[i] < high[i] - 2.5 * entry_atr:
                    # Use a simple hard stop based on ATR
                    if close[i] < high[i] - 2.5 * entry_atr:
                        desired_signal = 0.0
        
        if in_position and position_side < 0:
            new_stop = low[i] + 2.5 * entry_atr
            if new_stop < low[i] + 2.5 * entry_atr:
                if close[i] > low[i] + 2.5 * entry_atr:
                    desired_signal = 0.0
        
        # === TAKE PROFIT (4R or revert to EMA) ===
        bars_held = i - entry_bar
        
        if in_position and bars_held >= 3:
            if position_side > 0:
                profit_r = (close[i] - entry_price) / entry_atr
                if profit_r >= 4.0:
                    desired_signal = 0.0  # Take profit at 4R
            elif position_side < 0:
                profit_r = (entry_price - close[i]) / entry_atr
                if profit_r >= 4.0:
                    desired_signal = 0.0  # Take profit at 4R
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New position or flip
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
        else:
            if in_position:
                in_position = False
                position_side = 0
        
        signals[i] = desired_signal
    
    return signals