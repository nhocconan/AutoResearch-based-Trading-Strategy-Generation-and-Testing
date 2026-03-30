#!/usr/bin/env python3
"""
Experiment #008: 12h Donchian Breakout + Weekly Trend + Volume

HYPOTHESIS: 12h Donchian(20) breakout captures multi-day institutional moves.
Weekly EMA50 provides macro trend filter to avoid countertrend trades.
Volume confirms breakout validity.

WHY 12h: ~3x fewer trades than 4h = less fee drag.
Donchian(20) on 12h = 10-day channel - captures big directional moves.

WHY IT WORKS IN BOTH MARKETS:
- Bull: Breakout above Donchian high + weekly trend = continuation trades
- Bear: Breakout below Donchian low + weekly trend = short positions
- Symmetrical and regime-adaptive

TARGET: 75-150 total trades over 4 years (19-37/year).
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_vol_1w_ema50_v1"
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
    
    # Weekly EMA50 for macro trend filter
    ema_1w = pd.Series(df_1w['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # === Local 12h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Donchian channel (20 bars = 10 days on 12h)
    donchian_period = 20
    donchian_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # Volume ratio (20-bar moving average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
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
    
    warmup = max(100, donchian_period)  # Need enough for Donchian + EMA alignment buffer
    
    for i in range(warmup, n):
        # Skip if ATR not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # Skip if Weekly EMA not aligned
        if np.isnan(ema_1w_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === MACRO TREND (1w EMA50) ===
        price_above_1w_ema = close[i] > ema_1w_aligned[i]
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === DONCHIAN BREAKOUT detection (from CLOSED bars only - no look-ahead) ===
        # Previous bar's Donchian levels (shift by 1 to use only closed bars)
        prev_donchian_high = donchian_high[i - 1]
        prev_donchian_low = donchian_low[i - 1]
        
        # Breakout: current close breaks above/below previous Donchian channel
        breakout_up = close[i] > prev_donchian_high
        breakout_down = close[i] < prev_donchian_low
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG: Breakout above Donchian high + weekly trend up + volume ===
            if breakout_up and price_above_1w_ema and vol_spike:
                desired_signal = SIZE
            
            # === SHORT: Breakout below Donchian low + weekly trend down + volume ===
            if breakout_down and not price_above_1w_ema and vol_spike:
                desired_signal = -SIZE
        
        # === TRAILING STOPLOSS (2.0 ATR) ===
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.0 * atr_14[i]
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.0 * atr_14[i]
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                desired_signal = 0.0
        
        # === MINIMUM HOLD (3 bars = 1.5 days to avoid whipsaw) ===
        bars_held = i - entry_bar
        if in_position and bars_held >= 3:
            # Price reverts to Donchian mid = exit
            donchian_mid = (prev_donchian_high + prev_donchian_low) / 2
            if position_side > 0 and close[i] <= donchian_mid:
                desired_signal = 0.0
            if position_side < 0 and close[i] >= donchian_mid:
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