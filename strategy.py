#!/usr/bin/env python3
"""
Experiment #021: 4h Donchian Breakout + HMA21 + Volume Confirmation

HYPOTHESIS: Price channels (Donchian) capture institutional breakout moments.
Adding HMA21 trend filter + volume spike prevents false breakouts in choppy markets.
ATR-based stops ensure exits during volatility expansion.

WHY 4h: Proven timeframe from DB (most top performers use 4h).
WHY DONCHIAN: Simple, objective entry — no parameters to overfit.
WHY HMA21: Faster than EMA200, more smoothing than SMA, proven trend signal.

Target: 75-200 total trades over 4 years (19-50/year).
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_hma_vol_atr_v1"
timeframe = "4h"
leverage = 1.0


def calculate_hma(values, period):
    """Hull Moving Average"""
    half = period // 2
    sqrt_n = int(np.sqrt(period))
    
    wma1 = pd.Series(values).rolling(window=half, min_periods=half).apply(
        lambda x: np.sum(x * np.arange(1, len(x) + 1)) / np.sum(np.arange(1, len(x) + 1)), raw=True
    ).values
    wma2 = pd.Series(values).rolling(window=period, min_periods=period).apply(
        lambda x: np.sum(x * np.arange(1, len(x) + 1)) / np.sum(np.arange(1, len(x) + 1)), raw=True, raw=True
    ).values
    
    # Fill NaN with original values for first elements
    wma1 = np.where(np.isnan(wma1), values, wma1)
    wma2 = np.where(np.isnan(wma2), values, wma2)
    
    hma_raw = 2 * wma1 - wma2
    hma = pd.Series(hma_raw).rolling(window=sqrt_n, min_periods=sqrt_n).apply(
        lambda x: np.sum(x * np.arange(1, len(x) + 1)) / np.sum(np.arange(1, len(x) + 1)), raw=True
    ).values
    
    return np.where(np.isnan(hma), hma_raw, hma)


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
    
    # 1d HMA21 for trend direction
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # === Local 4h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Donchian channels (20 periods = 5 days)
    donchian_period = 20
    donchian_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # Volume ratio (20-period average)
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
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_bar = 0
    
    warmup = 100  # Buffer for all indicators
    
    for i in range(warmup, n):
        # Skip if ATR not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # Skip if HMA not aligned
        if np.isnan(hma_1d_aligned[i]):
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
        
        # === TREND DIRECTION (1d HMA21) ===
        trend_up = close[i] > hma_1d_aligned[i]
        
        # Volume confirmation
        vol_spike = vol_ratio[i] > 1.5
        
        # Donchian breakout signals
        price_broke_high = close[i] > donchian_high[i]
        price_broke_low = close[i] < donchian_low[i]
        
        # ATR-based entry threshold (breakout must exceed 0.5 ATR to confirm)
        atr_threshold = 0.5 * atr_14[i]
        high_breakout = close[i] - donchian_high[i - 1] if i > 0 else 0
        low_breakout = donchian_low[i - 1] - close[i] if i > 0 else 0
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG: Breakout above Donchian high + trend up + volume ===
            if trend_up and vol_spike and price_broke_high and high_breakout > atr_threshold:
                desired_signal = SIZE
            
            # === SHORT: Breakdown below Donchian low + trend down + volume ===
            if not trend_up and vol_spike and price_broke_low and low_breakout > atr_threshold:
                desired_signal = -SIZE
        
        # === STOPLOSS (2.0 ATR trailing) ===
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.0 * atr_14[i]
            if low[i] < trailing_stop:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.0 * atr_14[i]
            if high[i] > trailing_stop:
                desired_signal = 0.0
        
        # === MINIMUM HOLD (2 bars = 8h to avoid churn) ===
        bars_held = i - entry_bar
        if in_position and bars_held < 2:
            if position_side > 0 and desired_signal == 0:
                desired_signal = SIZE  # Keep long until minimum hold
            if position_side < 0 and desired_signal == 0:
                desired_signal = -SIZE  # Keep short until minimum hold
        
        # === TAKE PROFIT (2:1 ratio, reduce to half) ===
        if in_position and bars_held >= 4:  # At least 16h held
            if position_side > 0:
                profit = close[i] - entry_price
                if profit > 2.0 * atr_14[i]:  # 2R profit
                    desired_signal = SIZE / 2  # Take partial profit
            if position_side < 0:
                profit = entry_price - close[i]
                if profit > 2.0 * atr_14[i]:  # 2R profit
                    desired_signal = -SIZE / 2  # Take partial profit
        
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