#!/usr/bin/env python3
"""
Experiment #026: 4h Donchian Breakout + Volume Confirm + ATR Stop

HYPOTHESIS: Simple = fewer parameters = better generalization.
Based on kept strategy mtf_4h_hma_donchian_vol_chop_v2 (Sharpe 0.356, 358 trades).
Remove HMA and chop filter, rely on Donchian structure alone.
Volume spike (1.5x) confirms institutional interest.
2.5x ATR stoploss provides adaptive risk management.

Entry: Donchian(20) high/low break + volume spike (1.5x)
Stop: 2.5x ATR from entry
Min hold: 3 bars (reduce fee churn)

EXPECTED: 150-250 trades over 4 years (37-62/year per symbol)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_vol_atr_v6"
timeframe = "4h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
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
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Donchian Channel(20)
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume average (20 bars)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # === Signals ===
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    
    warmup = 60  # Enough for Donchian20, ATR14
    
    for i in range(warmup, n):
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            continue
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === DONCHIAN BREAKOUT ===
        # Use PREVIOUS bar's channel (shift by 1 to avoid look-ahead)
        prev_donchian_high = donchian_upper[i-1] if i > 0 and not np.isnan(donchian_upper[i-1]) else np.nan
        prev_donchian_low = donchian_lower[i-1] if i > 0 and not np.isnan(donchian_lower[i-1]) else np.nan
        
        bullish_breakout = False
        bearish_breakout = False
        
        if not np.isnan(prev_donchian_high) and not np.isnan(prev_donchian_low):
            # Break previous high = bullish
            if high[i] > prev_donchian_high:
                bullish_breakout = True
            # Break previous low = bearish
            if low[i] < prev_donchian_low:
                bearish_breakout = True
        
        # === ENTRY LOGIC ===
        if not in_position:
            # LONG: Breakout above 20-bar high + volume spike
            if bullish_breakout and vol_spike:
                signals[i] = SIZE
                in_position = True
                position_side = 1
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
            
            # SHORT: Breakdown below 20-bar low + volume spike
            elif bearish_breakout and vol_spike:
                signals[i] = -SIZE
                in_position = True
                position_side = -1
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
        
        # === EXIT LOGIC ===
        else:
            # Min hold: 3 bars to reduce fee churn
            min_hold_passed = (i - entry_bar) >= 3
            
            if position_side > 0:
                # Stop loss: 2.5x ATR from entry
                stop_price = entry_price - 2.5 * entry_atr
                if low[i] < stop_price:
                    signals[i] = 0.0
                    in_position = False
                    position_side = 0
                elif min_hold_passed:
                    # Reverse signal on opposite breakout
                    if bearish_breakout and vol_spike:
                        signals[i] = -SIZE
                        in_position = True
                        position_side = -1
                        entry_price = close[i]
                        entry_atr = atr_14[i]
                        entry_bar = i
                    else:
                        signals[i] = SIZE
                else:
                    signals[i] = SIZE
            
            elif position_side < 0:
                # Stop loss: 2.5x ATR from entry
                stop_price = entry_price + 2.5 * entry_atr
                if high[i] > stop_price:
                    signals[i] = 0.0
                    in_position = False
                    position_side = 0
                elif min_hold_passed:
                    # Reverse signal on opposite breakout
                    if bullish_breakout and vol_spike:
                        signals[i] = SIZE
                        in_position = True
                        position_side = 1
                        entry_price = close[i]
                        entry_atr = atr_14[i]
                        entry_bar = i
                    else:
                        signals[i] = -SIZE
                else:
                    signals[i] = -SIZE
    
    return signals