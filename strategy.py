#!/usr/bin/env python3
"""
Experiment #028: 6h Elder Force Index + Donchian Breakout + ATR Stops

HYPOTHESIS: Elder's Force Index (EFI) measures institutional pressure by combining
price change with volume. Donchian breakout provides price structure. Together they
create a 2-factor entry: institutional momentum confirms structural break.

WHY GENUINELY DIFFERENT: Previous strategies used RSI, Williams %R, TRIX, ADX, KAMA,
HMA for momentum - but NONE used Force Index (price * volume). EFI captures the key
difference between retail-driven and institutional-driven moves.

WHY IT WORKS IN BOTH MARKETS:
- Bull: Long breakouts with positive EFI = institutional accumulation
- Bear: Short breakdowns with negative EFI = institutional distribution
- Low volume breakouts get filtered = avoids bull trap whipsaws

TARGET: 50-150 total trades over 4 years. Size: 0.25.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_efi_donchian_atr_v1"
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

def calculate_efi(close, volume, period=13):
    """Elder Force Index - measures institutional pressure"""
    n = len(close)
    if n < 2:
        return np.full(n, np.nan)
    
    # Raw force = price change * volume
    force = np.zeros(n, dtype=np.float64)
    force[1:] = (close[1:] - close[:-1]) * volume[1:]
    
    # Smooth with EMA
    efi = pd.Series(force).ewm(span=period, min_periods=period, adjust=False).mean().values
    return efi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Local 6h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    efi = calculate_efi(close, volume, period=13)
    
    # Donchian channel (20 periods = 5 days on 6h)
    donchian_period = 20
    donchian_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Volume SMA for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_bar = 0
    stop_price = 0.0
    
    warmup = 100  # Donchian(20) + buffer
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            position_side = 0
            continue
        
        if np.isnan(efi[i]):
            signals[i] = 0.0
            position_side = 0
            continue
        
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]):
            signals[i] = 0.0
            position_side = 0
            continue
        
        # === REFERENCE VALUES (use PREVIOUS bar to avoid look-ahead) ===
        prev_donchian_high = donchian_high[i - 1]
        prev_donchian_low = donchian_low[i - 1]
        prev_efi = efi[i - 1]
        current_high = high[i]
        current_low = low[i]
        
        # === EFI MOMENTUM DIRECTION ===
        efi_positive = efi[i] > 0
        efi_negative = efi[i] < 0
        efi_improving = efi[i] > prev_efi  # Momentum accelerating
        
        # === VOLUME CONFIRMATION ===
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 1.0
        vol_above_avg = vol_ratio > 1.0  # At least average volume
        
        # === ENTRY CONDITIONS ===
        desired_signal = 0.0
        
        if position_side == 0:
            # === LONG: Breakout above Donchian high + positive EFI ===
            if current_high > prev_donchian_high:
                if efi_positive and (efi_improving or vol_above_avg):
                    desired_signal = SIZE
            
            # === SHORT: Breakdown below Donchian low + negative EFI ===
            elif current_low < prev_donchian_low:
                if efi_negative and (not efi_improving or vol_above_avg):
                    desired_signal = -SIZE
        
        # === STOPLOSS (2.0 ATR from entry or trailing) ===
        if position_side != 0:
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                trailing_stop = highest_since_entry - 2.0 * entry_atr
                stop_price = max(stop_price, trailing_stop)
                if low[i] < stop_price:
                    desired_signal = 0.0
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                trailing_stop = lowest_since_entry + 2.0 * entry_atr
                stop_price = min(stop_price, trailing_stop)
                if high[i] > stop_price:
                    desired_signal = 0.0
        
        # === TIME EXIT (hold at least 8 bars = 2 days) ===
        bars_held = i - entry_bar
        if position_side != 0 and bars_held >= 8:
            if position_side > 0 and close[i] < donchian_mid[i]:
                desired_signal = 0.0
            if position_side < 0 and close[i] > donchian_mid[i]:
                desired_signal = 0.0
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if position_side == 0 or np.sign(desired_signal) != position_side:
                # New position or flip
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
            if position_side != 0:
                position_side = 0
        
        signals[i] = desired_signal
    
    return signals