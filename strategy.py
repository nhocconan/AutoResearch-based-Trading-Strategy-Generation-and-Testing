#!/usr/bin/env python3
"""
Experiment #006: 4h TRIX Momentum + Donchian Breakout + Volume Confirmation

HYPOTHESIS: TRIX (Triple EMA Oscillator) catches momentum shifts better than
single EMA crosses. Combined with Donchian(20) breakout for structure and
volume confirmation, this catches trend changes at key levels.

WHY IT WORKS IN BOTH BULL AND BEAR: TRIX oscillates around zero - long when
crossing above zero, short when crossing below. Symmetric entry. Works in
both uptrends and downtrends.

TIMING: TRIX crossover is a proven reversal signal. Donchian(20) on 4h = 80h
(~3 days) breakout captures multi-day swings. Volume confirmation filters
false breakouts.

TARGET: 75-200 total trades over 4 years (19-50/year).
Signal size: 0.25. ATR stoploss at 2.5x.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_trix_donchian_vol_1d_v1"
timeframe = "4h"
leverage = 1.0

def calculate_trix(close, period=14):
    """Triple EMA Oscillator - momentum indicator"""
    n = len(close)
    if n < period * 3:
        return np.full(n, np.nan)
    
    ema1 = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    ema2 = pd.Series(ema1).ewm(span=period, min_periods=period, adjust=False).mean().values
    ema3 = pd.Series(ema2).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # TRIX = rate of change of triple EMA
    trix = np.zeros(n)
    trix[period*2:] = (ema3[period*2:] - ema3[period*2-1:period*2-2:-1]) / ema3[period*2-1:period*2-2:-1] * 100
    
    return trix

def calculate_donchian(high, low, period=20):
    """Donchian Channel - return upper and lower bands"""
    n = len(high)
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

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
    
    # 1d EMA50 for trend direction
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # === Local 4h indicators ===
    # TRIX(14) - momentum oscillator
    trix = calculate_trix(close, period=14)
    
    # TRIX signal line (EMA of TRIX)
    trix_ema = pd.Series(trix).ewm(span=9, min_periods=9, adjust=False).mean().values
    
    # Donchian(20) - 4h price structure
    donch_upper, donch_lower = calculate_donchian(high, low, period=20)
    
    # ATR for stoploss
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Volume ratio (20-period MA)
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
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_bar = 0
    prev_trix_above = False
    
    warmup = 200  # Need enough for TRIX + EMA alignment
    
    for i in range(warmup, n):
        # Skip if ATR not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # Skip if indicators not aligned
        if np.isnan(ema_1d_aligned[i]) or np.isnan(trix[i]) or np.isnan(trix_ema[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # Skip if Donchian not ready
        if np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === TREND DIRECTION (1d EMA50) ===
        price_above_1d_ema = close[i] > ema_1d_aligned[i]
        price_below_1d_ema = close[i] < ema_1d_aligned[i]
        
        # === TRIX CROSSOVER (momentum shift) ===
        trix_above_signal = trix[i] > trix_ema[i]
        prev_trix_above = trix_above_signal if i == warmup or np.isnan(trix[i-1]) else prev_trix_above
        
        # Detect crossover (not crossunder) - need actual previous state
        if not np.isnan(trix[i-1]) and not np.isnan(trix_ema[i-1]):
            prev_trix_above = trix[i-1] > trix_ema[i-1]
        
        trix_cross_up = (not prev_trix_above) and trix_above_signal
        trix_cross_down = prev_trix_above and (not trix_above_signal)
        
        # === DONCHIAN BREAKOUT ===
        # Upper breakout: price breaks above 20-period high
        upper_breakout = close[i] > donch_upper[i] and close[i-1] <= donch_upper[i-1] if i > 0 else False
        # Lower breakout: price breaks below 20-period low
        lower_breakout = close[i] < donch_lower[i] and close[i-1] >= donch_lower[i-1] if i > 0 else False
        
        # Volume confirmation
        vol_spike = vol_ratio[i] > 1.5
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG: TRIX crosses up + price above 1d EMA + (upper breakout OR vol spike) ===
            if trix_cross_up and price_above_1d_ema:
                # Require either upper Donchian breakout OR strong volume
                if upper_breakout or (vol_spike and close[i] > donch_lower[i] + 0.5 * (donch_upper[i] - donch_lower[i])):
                    desired_signal = SIZE
            
            # === SHORT: TRIX crosses down + price below 1d EMA + (lower breakout OR vol spike) ===
            if trix_cross_down and price_below_1d_ema:
                if lower_breakout or (vol_spike and close[i] < donch_lower[i] + 0.5 * (donch_upper[i] - donch_lower[i])):
                    desired_signal = -SIZE
        
        # === STOPLOSS (2.5 ATR trailing) ===
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            if low[i] < trailing_stop:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            if high[i] > trailing_stop:
                desired_signal = 0.0
        
        # === MINIMUM HOLD = 3 BARS (12h) to avoid churn ===
        bars_held = i - entry_bar
        if in_position and bars_held < 3:
            # Don't exit early, just hold
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
        else:
            if in_position:
                in_position = False
                position_side = 0
        
        signals[i] = desired_signal
    
    return signals