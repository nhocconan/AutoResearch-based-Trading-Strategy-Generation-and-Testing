#!/usr/bin/env python3
"""
Experiment #008: 12h Weekly EMA Trend + 12-bar Donchian + Volume Spike

HYPOTHESIS: 1w EMA crossover identifies major trend changes. 12-bar (6-day) 
Donchian breakout on 12h captures multi-day momentum moves. Volume spike 
confirms institutional participation. ATR stop provides risk control.

WHY IT WORKS IN BULL AND BEAR: Symmetric breakout strategy - long breakouts
above 1w EMA in uptrends, short breakouts below 1w EMA in downtrends.

TARGET: 75-125 total trades over 4 years. Signal size: 0.30.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_weekly_ema_donchian_vol_1w_v1"
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
    
    # 1w EMA21 vs EMA50 for trend direction
    ema_21_1w = pd.Series(df_1w['close'].values).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_21_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # === Local 12h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Volume ratio (20-bar)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # 12-bar Donchian (6-day channel)
    donchian_period = 12
    donchian_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_bar = 0
    
    warmup = 300  # Need enough for 1w EMA50 alignment
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(ema_21_aligned[i]) or np.isnan(ema_50_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]):
            signals[i] = 0.0
            continue
        
        # === TREND DIRECTION (1w EMA21 vs EMA50) ===
        weekly_uptrend = ema_21_aligned[i] > ema_50_aligned[i]
        weekly_downtrend = ema_21_aligned[i] < ema_50_aligned[i]
        
        # Volume confirmation
        vol_spike = vol_ratio[i] > 1.5
        
        # === DONCHIAN BREAKOUT detection ===
        # Upper breakout: close above 12-bar high
        upper_breakout = close[i] > donchian_high[i - 1] if i > donchian_period else False
        # Lower breakout: close below 12-bar low
        lower_breakout = close[i] < donchian_low[i - 1] if i > donchian_period else False
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG: Upper Donchian breakout + 1w uptrend + volume ===
            if upper_breakout and weekly_uptrend and vol_spike:
                desired_signal = SIZE
            
            # === SHORT: Lower Donchian breakout + 1w downtrend + volume ===
            if lower_breakout and weekly_downtrend and vol_spike:
                desired_signal = -SIZE
        
        # === TRAILING STOP (2.0 ATR) ===
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
        
        # === HOLD PERIOD (minimum 3 bars to reduce churn) ===
        bars_held = i - entry_bar
        
        # === REVERSAL: Opposite breakout while in position ===
        if in_position and bars_held >= 3:
            # If we go long but now get short signal with opposite trend
            if position_side > 0 and lower_breakout and weekly_downtrend and vol_spike:
                desired_signal = 0.0
            if position_side < 0 and upper_breakout and weekly_uptrend and vol_spike:
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