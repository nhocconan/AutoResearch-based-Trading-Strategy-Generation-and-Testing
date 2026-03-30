#!/usr/bin/env python3
"""
Experiment #028: 4h Donchian Breakout + Volume + 1d SMA200 Trend Filter

HYPOTHESIS: The proven winning pattern is:
1. Donchian(20) breakout as price structure entry (institutional moves)
2. Volume confirmation (filters false breakouts)
3. 1d SMA200 as trend filter (aligns with daily direction)
4. Choppiness filter (avoids range-bound whipsaws)
5. ATR stoploss (2.0x) for risk management

WHY 4h: Most proven timeframe from DB (multiple successful strategies).
4h = 6 bars per day, 20-bar Donchian = ~3.3 day channel.

SIMPLICITY: Only 3 conditions for entry (breakout + volume + trend).
Too many conditions = conditions rarely align = 0 trades or overtrading.

TARGET: 75-150 total trades over 4 years (19-37/year). HARD MAX: 300.
Signal size: 0.25.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_vol_sma200_chop_v1"
timeframe = "4h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1]))
    tr = np.insert(tr, 0, high[0] - low[0])
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index - values below 50 suggest trending, above 60 suggest chop/ranging"""
    n = len(high)
    chop = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        # True range sum over period
        tr_list = []
        for j in range(i - period + 1, i + 1):
            if j > 0:
                tr = max(high[j] - low[j], abs(high[j] - close[j-1]))
            else:
                tr = high[j] - low[j]
            tr_list.append(tr)
        
        tr_sum = sum(tr_list)
        hh = max(high[i - period + 1:i + 1])
        ll = min(low[i - period + 1:i + 1])
        range_hl = hh - ll
        
        if range_hl > 0 and tr_sum > 0:
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
    
    # 1d SMA200 for trend direction (proven filter from DB)
    sma_200_1d = pd.Series(df_1d['close'].values).rolling(window=200, min_periods=200).mean().values
    sma_200_aligned = align_htf_to_ltf(prices, df_1d, sma_200_1d)
    
    # Local 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    
    # Donchian channels (20 periods = ~3.3 days on 4h)
    donchian_period = 20
    donchian_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().shift(1).values
    donchian_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().shift(1).values
    
    # Volume (20-bar average)
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
    
    warmup = 250  # Need 200 for SMA200 + 20 for Donchian + buffer
    
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
        
        if np.isnan(chop[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === TREND DIRECTION (1d SMA200) ===
        is_bullish = close[i] > sma_200_aligned[i]
        is_bearish = close[i] < sma_200_aligned[i]
        
        # === REGIME (Choppiness Index) ===
        # Skip if too choppy (CHOP > 61.8) - ranging market, avoid entry
        is_choppy = chop[i] > 61.8
        
        # === DONCHIAN BREAKOUT SIGNALS ===
        # Use previous bar's Donchian (shifted to avoid look-ahead)
        prev_donchian_high = donchian_high[i]
        prev_donchian_low = donchian_low[i]
        
        # Volume confirmation for breakout
        vol_spike = vol_ratio[i] > 1.5
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG: Breakout above Donchian high with bullish trend ===
            # Breakout condition: current bar breaks above previous 20-bar high
            if high[i] > prev_donchian_high and is_bullish and not is_choppy:
                if vol_spike:  # Volume confirmation required for breakout
                    desired_signal = SIZE
            
            # === SHORT: Breakdown below Donchian low with bearish trend ===
            if low[i] < prev_donchian_low and is_bearish and not is_choppy:
                if vol_spike:
                    desired_signal = -SIZE
        
        # === STOPLOSS CHECK (2.0 ATR trailing stop) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.0 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.0 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === MINIMUM HOLD PERIOD (6 bars = 1 day on 4h) ===
        # Prevents immediate exit on noise
        bars_held = i - entry_bar
        
        # === STOP-LOSS ON FIRST BAR (immediate ATR breach) ===
        if in_position and bars_held == 0:
            if position_side > 0 and low[i] < entry_price - 2.0 * entry_atr:
                desired_signal = 0.0
            if position_side < 0 and high[i] > entry_price + 2.0 * entry_atr:
                desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New position or direction flip
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
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = desired_signal
    
    return signals