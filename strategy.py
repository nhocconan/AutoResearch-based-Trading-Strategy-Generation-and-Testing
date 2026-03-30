#!/usr/bin/env python3
"""
Experiment #025: 4h Donchian + HTF Trend (Minimalist)

HYPOTHESIS: Most failed strategies have TOO MANY overlapping entry conditions.
Best performers (Sharpe 1.31-1.47) use ONE strong signal + 1-2 filters.
Strategy: Donchian(20) breakout as ONLY signal, HTF 1d EMA50 as ONLY filter.
Volume confirmation for entries. ATR stops.

WHY IT WORKS IN BULL + BEAR:
- Bull: Breakout above Donchian upper + price > 1d EMA50 = momentum continuation
- Bear: Breakdown below Donchian lower + price < 1d EMA50 = short continuation
- Range: No trades (breakouts fail in chop = natural filter)

TARGET: 75-150 total trades over 4 years (19-37/year on 4h).
Previous TRIX+Camarilla: 172 trades with -0.215 Sharpe (too many conditions).
This version: 1 signal + 1 filter = tighter entries.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_htf_ema_minimal_v1"
timeframe = "4h"
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
    
    # 1d EMA50 for trend direction
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === 4h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Donchian Channel (20 periods = 5 days on 4h)
    upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: 20-bar MA ratio
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
    entry_bar = 0
    
    warmup = 100  # Need 20 for Donchian + 50 for HTF EMA + buffer
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(ema_50_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === HTF TREND: Primary filter ===
        above_htf_ema = close[i] > ema_50_aligned[i]
        below_htf_ema = close[i] < ema_50_aligned[i]
        
        # === VOLUME CONFIRMATION ===
        vol_confirm = vol_ratio[i] > 1.5
        
        # === DONCHIAN BREAKOUT (shift by 1 to avoid look-ahead) ===
        donchian_up_prev = upper[i - 1]
        donchian_lo_prev = lower[i - 1]
        
        breakout_up = close[i] > donchian_up_prev
        breakout_down = close[i] < donchian_lo_prev
        
        # === MINIMUM HOLD (3 bars to avoid fee churn) ===
        bars_held = i - entry_bar if in_position else 999
        
        # === ENTRY LOGIC: SINGLE signal + SINGLE filter ===
        desired_signal = 0.0
        
        if not in_position:
            # LONG: Breakout above Donchian upper + above HTF EMA + volume
            if breakout_up and above_htf_ema and vol_confirm:
                desired_signal = SIZE
            
            # SHORT: Breakdown below Donchian lower + below HTF EMA + volume
            if breakout_down and below_htf_ema and vol_confirm:
                desired_signal = -SIZE
        
        # === EXIT LOGIC ===
        if in_position:
            # Stop-loss: price moves 3 ATR against position
            if position_side > 0:
                stop_price = entry_price - 3.0 * entry_atr
                if low[i] < stop_price:
                    desired_signal = 0.0
            
            elif position_side < 0:
                stop_price = entry_price + 3.0 * entry_atr
                if high[i] > stop_price:
                    desired_signal = 0.0
            
            # Exit on opposite Donchian band (with minimum hold)
            if bars_held >= 3:
                if position_side > 0 and breakout_down:
                    desired_signal = 0.0
                elif position_side < 0 and breakout_up:
                    desired_signal = 0.0
        
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