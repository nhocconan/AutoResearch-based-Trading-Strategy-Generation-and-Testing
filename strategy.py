#!/usr/bin/env python3
"""
Experiment #008: 12h Camarilla + Weekly ATR Regime + Volume

HYPOTHESIS: Camarilla S3/S4 and R3/R4 are institutional support/resistance levels.
By requiring TRIPLE confirmation (Camarilla level touch + volume spike + ATR regime),
entries become rare enough for 12h timeframe (50-150 trades/4yr).

WHY IT WORKS IN BOTH BULL AND BEAR:
- Bull: Buy S3/S4 touches when price > weekly EMA50 (buy the dip)
- Bear: Short R3/R4 touches when price < weekly EMA50 (sell the rally)
- ATR regime ensures we only enter during volatility expansion (momentum)

KEY DIFFERENCE from failed #020 (280 trades):
- #020 required ONLY Camarilla + vol OR EMA (too loose)
- This version requires Camarilla + vol AND ATR regime (all three = ~70% fewer trades)

TARGET: 75-150 total over 4 years = 19-37/year. Signal size: 0.25.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_camarilla_weekly_atr_vol_v1"
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
    
    # Weekly EMA50 for trend (captures multi-week direction)
    ema_1w = pd.Series(df_1w['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # === Local 12h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    atr_ma = pd.Series(atr_14).rolling(window=20, min_periods=20).mean().values
    atr_ratio = atr_14 / np.where(atr_ma > 0, atr_ma, 1)
    
    # Volume ratio (20-bar MA)
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
    
    warmup = 100
    
    for i in range(warmup, n):
        # Skip if ATR not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # Skip if weekly EMA not aligned
        if np.isnan(ema_1w_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === FILTERS ===
        price_above_1w_ema = close[i] > ema_1w_aligned[i]
        vol_spike = vol_ratio[i] > 1.8
        atr_expanding = atr_ratio[i] > 1.15  # ATR regime: volatility expanding
        
        # === CAMARILLA LEVELS from previous CLOSED bar ===
        prev_high = high[i - 1]
        prev_low = low[i - 1]
        prev_close = close[i - 1]
        prev_range = prev_high - prev_low
        
        # Classic Camarilla levels
        r3 = prev_close + prev_range * 0.09167
        r4 = prev_close + prev_range * 0.18333
        s3 = prev_close - prev_range * 0.09167
        s4 = prev_close - prev_range * 0.18333
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # LONG: Price touches S3/S4 + bullish trend + volume + ATR regime
            if price_above_1w_ema and vol_spike and atr_expanding:
                if low[i] <= s4:
                    desired_signal = SIZE
                elif low[i] <= s3:
                    desired_signal = SIZE
            
            # SHORT: Price touches R3/R4 + bearish trend + volume + ATR regime
            if not price_above_1w_ema and vol_spike and atr_expanding:
                if high[i] >= r4:
                    desired_signal = -SIZE
                elif high[i] >= r3:
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
        
        # === MINIMUM HOLD (4 bars = 2 days) ===
        bars_held = i - entry_bar
        
        # === EXIT on ATR compression (regime ends) ===
        if in_position and bars_held >= 4:
            # Exit long when ATR compresses (trend exhausted)
            if position_side > 0 and atr_ratio[i] < 1.1:
                desired_signal = 0.0
            # Exit short when ATR compresses
            if position_side < 0 and atr_ratio[i] < 1.1:
                desired_signal = 0.0
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                entry_bar = i
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                stop_price = 0.0
        
        signals[i] = desired_signal
    
    return signals