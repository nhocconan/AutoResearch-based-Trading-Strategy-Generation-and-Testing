#!/usr/bin/env python3
"""
Experiment #006: 4h Donchian Breakout + Volume Confirmation + 1d HMA Trend

HYPOTHESIS: Donchian(20) breakout is a proven institutional structure pattern.
By entering when price breaks above/below the 4h Donchian channel WITH volume 
confirmation AND 1d HMA trend alignment, this captures momentum bursts.

WHY IT WORKS IN BOTH BULL AND BEAR:
- Bull: Buy breakouts above 1d HMA (trend-following)
- Bear: Short breakouts below 1d HMA (momentum continuation)
- Symmetrical: Same logic in both directions

WHY 4h: 4h captures multi-day trends without overtrading (vs 15m/30m).
Target: 100-200 total trades over 4 years. HARD MAX: 300.

KEY INSIGHT FROM DB: Top performers use tight entries (Donchian breakout + volume),
not loose entries. Loose entries (>400 trades) always fail due to fee drag.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_vol_hma_1d_v1"
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

def calculate_hma(values, period):
    """Hull Moving Average"""
    half = period // 2
    sqrt_n = int(np.sqrt(period))
    
    wma_half = pd.Series(values).rolling(window=half, min_periods=half).mean()
    wma_full = pd.Series(values).rolling(window=period, min_periods=period).mean()
    
    diff = 2 * wma_half - wma_full
    hma = diff.rolling(window=sqrt_n, min_periods=sqrt_n).mean()
    
    return hma.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d HMA16 for trend direction (faster than EMA200)
    hma_1d = calculate_hma(df_1d['close'].values, 16)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # === Local 4h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Donchian Channel (20 periods = 80 hours = 3.3 days)
    donchian_period = 20
    donchian_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # Volume ratio (20 period MA)
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
    
    warmup = max(100, donchian_period + 20)  # Need enough for Donchian + alignment
    
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
        
        # Get previous bar's Donchian (no look-ahead)
        prev_donchian_high = donchian_high[i - 1]
        prev_donchian_low = donchian_low[i - 1]
        
        # === TREND DIRECTION (1d HMA16) ===
        price_above_1d_hma = close[i] > hma_1d_aligned[i]
        
        # Volume confirmation (1.3x = moderate, not too strict)
        vol_spike = vol_ratio[i] > 1.3
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG: Break above Donchian high with volume + trend alignment ===
            if price_above_1d_hma and vol_spike:
                # Price breaks above previous Donchian high
                if close[i] > prev_donchian_high:
                    desired_signal = SIZE
            
            # === SHORT: Break below Donchian low with volume + trend alignment ===
            if not price_above_1d_hma and vol_spike:
                # Price breaks below previous Donchian low
                if close[i] < prev_donchian_low:
                    desired_signal = -SIZE
        
        # === STOPLOSS (2.0 ATR trailing) ===
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
        
        # === MINIMUM HOLD: 3 bars (12 hours) to avoid churn ===
        bars_held = i - entry_bar
        
        if in_position and bars_held >= 3:
            # Exit on trend reversal (price crosses 1d HMA)
            if position_side > 0 and close[i] < hma_1d_aligned[i]:
                desired_signal = 0.0
            if position_side < 0 and close[i] > hma_1d_aligned[i]:
                desired_signal = 0.0
        
        # === OPPOSITE SIGNAL FLIP ===
        if in_position and bars_held >= 3:
            if position_side > 0 and not price_above_1d_hma and vol_spike:
                # Trend flipped to bearish
                if close[i] < prev_donchian_low:
                    desired_signal = -SIZE
            if position_side < 0 and price_above_1d_hma and vol_spike:
                # Trend flipped to bullish
                if close[i] > prev_donchian_high:
                    desired_signal = SIZE
        
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