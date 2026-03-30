#!/usr/bin/env python3
"""
Experiment #021: 4h Donchian(20) + KAMA Trend + Volume

HYPOTHESIS: Donchian(20) breakouts are institutional momentum signals.
KAMA trend filter ensures entries align with multi-day trend direction.
Volume confirms institutional participation.
ATR stoploss manages risk.

TRADE COUNT ESTIMATE:
- Donchian(20) on 4h: ~20-25 breakouts/year
- KAMA + volume filter: ~50% pass = 10-12 signals/year
- 4 years = 40-48 signals × 2 (long/short) ≈ 80-120 total trades
- Within target range (75-200).

WHY IT WORKS IN BOTH BULL AND BEAR:
- Bull: price breaks above 20-high, KAMA up = strong continuation
- Bear: price breaks below 20-low, KAMA down = momentum shorts
- Works on rallies and crashes alike.

KEY DIFFERENCES FROM FAILED ATTEMPTS:
- Uses KAMA (proven in DB winner) instead of HMA/EMA200
- Minimal conditions (3) to avoid overtrading
- No Williams %R (consistently failed in history)
- Proper stoploss via trailing ATR
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_kama_volume_1d_v1"
timeframe = "4h"
leverage = 1.0


def calculate_kama(close, period=10, fast_ema=2, slow_ema=30):
    """Kaufman's Adaptive Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    # Calculate Efficiency Ratio (ER)
    direction = np.abs(close[period:] - close[:-period])
    volatility = np.abs(np.diff(close, prepend=close[0]))
    rolling_vol = pd.Series(volatility).rolling(window=period, min_periods=period).sum()
    er = np.zeros(n)
    er[period:] = direction / np.where(rolling_vol > 0, rolling_vol, 1)
    er[:period] = er[period] if period < n else 0
    
    # Calculate smoothing constant
    fast_const = 2 / (fast_ema + 1)
    slow_const = 2 / (slow_ema + 1)
    sc = (er * (fast_const - slow_const) + slow_const) ** 2
    
    # KAMA calculation
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama


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


def calculate_donchian(high, low, period=20):
    """Donchian Channel - 20 period is standard"""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).max().values  # Note: min for lower
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    middle = (upper + lower) / 2
    return upper, middle, lower


def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d KAMA for trend direction (faster adaptation than EMA)
    kama_1d = calculate_kama(df_1d['close'].values, period=10)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    
    # === Local 4h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Donchian(20) channel
    donch_upper, donch_mid, donch_lower = calculate_donchian(high, low, period=20)
    
    # Volume ratio
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # === Signals ===
    signals = np.zeros(n)
    SIZE = 0.30  # Discrete position size
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_bar = 0
    
    warmup = 100  # Buffer for KAMA alignment
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(kama_1d_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === TREND DIRECTION (1d KAMA) ===
        price_above_kama = close[i] > kama_1d_aligned[i]
        price_below_kama = close[i] < kama_1d_aligned[i]
        
        # Volume confirmation
        vol_spike = vol_ratio[i] > 1.5
        
        # === DONCHIAN CHANNEL LEVELS ===
        upper = donch_upper[i]
        lower = donch_lower[i]
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG: Price breaks above Donchian upper with trend + volume ===
            if price_above_kama and vol_spike:
                if close[i] > upper:
                    desired_signal = SIZE
            
            # === SHORT: Price breaks below Donchian lower with trend + volume ===
            if price_below_kama and vol_spike:
                if close[i] < lower:
                    desired_signal = -SIZE
        
        # === TRAILING STOPLOSS (2.0 ATR) ===
        if in_position:
            if position_side > 0:
                highest_since_entry = max(highest_since_entry, high[i])
                trailing_stop = highest_since_entry - 2.0 * entry_atr
                stop_price = max(stop_price, trailing_stop)
                if low[i] < stop_price:
                    desired_signal = 0.0
            
            if position_side < 0:
                lowest_since_entry = min(lowest_since_entry, low[i])
                trailing_stop = lowest_since_entry + 2.0 * entry_atr
                stop_price = min(stop_price, trailing_stop)
                if high[i] > stop_price:
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