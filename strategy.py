#!/usr/bin/env python3
"""
Experiment #007: 6h Donchian Breakout + Weekly ATR Regime Filter

HYPOTHESIS: 
- Donchian(20) on 6h catches major trend changes (20*6h = 5 days)
- Weekly ATR regime determines breakout behavior:
  - Trending (ATR_1w > SMA_1w ATR): trade breakouts
  - Range (ATR_1w < SMA_1w ATR): fade breakouts at mid-channel
- Volume confirmation prevents false breakouts
- Works in BOTH bull and bear: long breakouts when bullish, short when bearish

WHY 6h: ~14600 bars over 4 years, 100 trades = 1 per 146 bars = selective entries
TARGET: 100-200 total trades over 4 years = 25-50/year. HARD MAX: 300.
Signal size: 0.30.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_donchian_weekly_atr_regime_v1"
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
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly ATR for regime detection
    h_1w = df_1w['high'].values
    l_1w = df_1w['low'].values
    c_1w = df_1w['close'].values
    atr_1w = calculate_atr(h_1w, l_1w, c_1w, period=14)
    atr_1w_sma = pd.Series(atr_1w).rolling(window=4, min_periods=2).mean().values
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    atr_1w_sma_aligned = align_htf_to_ltf(prices, df_1w, atr_1w_sma)
    
    # === Local 6h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Donchian channels (20 bars = 5 days)
    donchian_period = 20
    upper = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    lower = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    mid = (upper + lower) / 2.0
    
    # Volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_bar = 0
    
    warmup = max(100, donchian_period + 20)
    
    for i in range(warmup, n):
        # Skip if ATR not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # Skip if weekly ATR not aligned
        if np.isnan(atr_1w_aligned[i]) or np.isnan(atr_1w_sma_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === REGIME DETECTION (Weekly ATR) ===
        atr_ratio = atr_1w_aligned[i] / (atr_1w_sma_aligned[i] + 1e-10)
        is_trending = atr_ratio > 1.0  # ATR expanding = trending
        
        # Volume confirmation
        vol_spike = vol_ratio[i] > 1.5
        
        # Donchian levels
        d_upper = upper[i]
        d_lower = lower[i]
        d_mid = mid[i]
        
        # Previous bar close for mid-channel reference
        prev_close = close[i - 1]
        
        desired_signal = 0.0
        
        # === ENTRY LOGIC ===
        if not in_position:
            # === BREAKOUT LONG ===
            # Price breaks above Donchian upper
            if close[i] > d_upper and vol_spike:
                if is_trending:
                    # In trending market: ride the breakout
                    desired_signal = SIZE
                else:
                    # In range market: wait for pullback to mid
                    if low[i] <= d_mid:
                        desired_signal = SIZE
            
            # === BREAKOUT SHORT ===
            # Price breaks below Donchian lower
            if close[i] < d_lower and vol_spike:
                if is_trending:
                    # In trending market: ride the breakdown
                    desired_signal = -SIZE
                else:
                    # In range market: wait for rally to mid
                    if high[i] >= d_mid:
                        desired_signal = -SIZE
        
        # === STOPLOSS (2.5 ATR trailing) ===
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
        
        # === TAKE PROFIT: Revert to mid-channel in range market ===
        bars_held = i - entry_bar
        if in_position and bars_held >= 3 and not is_trending:
            if position_side > 0 and close[i] >= d_mid:
                desired_signal = 0.0
            if position_side < 0 and close[i] <= d_mid:
                desired_signal = 0.0
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                entry_bar = i
                if position_side > 0:
                    stop_price = low[i] - 2.5 * entry_atr
                else:
                    stop_price = high[i] + 2.5 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                stop_price = 0.0
        
        signals[i] = desired_signal
    
    return signals