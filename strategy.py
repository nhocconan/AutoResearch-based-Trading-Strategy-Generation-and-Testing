#!/usr/bin/env python3
"""
Experiment #006: 4h Camarilla Pivot + Volume + Choppiness Regime

HYPOTHESIS: Camarilla levels work best as MEAN REVERSION signals in RANGING
markets (CHOP > 61.8). Buy S3/S4 bounces in uptrends, short R3/R4 in downtrends.

WHY 4h: Better trade frequency than 12h while maintaining structural levels.
- Camarilla levels on 4h capture 6h-24h oscillations
- Choppiness(14) > 61.8 = ranging = Camarilla reversals work
- Volume spike confirms institutional interest at levels

CRITICAL DIFFERENCE FROM FAILED STRATS:
- DB winner uses CHOPPYNESS regime (not just EMA trend) — this is the key filter
- Many failed experiments lacked choppiness or had it wrong

TARGET: 75-150 total trades over 4 years (19-37/year)
Signal size: 0.30
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_camarilla_vol_chop_1d_v1"
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

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CEGISO)
    CHOP > 61.8 = ranging (good for mean reversion / Camarilla levels)
    CHOP < 38.2 = trending (avoid Camarilla reversals)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    for i in range(period, n):
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        
        if highest_high - lowest_low > 1e-10:
            sum_tr = 0.0
            for j in range(period):
                tr_cur = max(high[i - j] - low[i - j], 
                            abs(high[i - j] - close[i - j - 1] if i - j - 1 >= 0 else high[i - j] - low[i - j]),
                            abs(low[i - j] - close[i - j - 1] if i - j - 1 >= 0 else high[i - j] - low[i - j]))
                sum_tr += tr_cur
            
            chop[i] = 100 * (np.log(sum_tr) / np.log(highest_high - lowest_low)) if (highest_high - lowest_low) > 1e-10 else 50.0
    
    return chop

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
    atr_14 = calculate_atr(high, low, close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    
    # Volume ratio (20-bar MA)
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
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_bar = 0
    last_entry_level = 0  # Track which Camarilla level we entered at
    bars_since_entry = 0
    
    warmup = 100
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(chop_14[i]) or np.isnan(ema_1d_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === REGIME CHECK ===
        is_choppy = chop_14[i] > 61.8  # Only trade Camarilla reversals in ranging markets
        
        # Volume confirmation
        vol_spike = vol_ratio[i] > 1.8
        
        # === TREND DIRECTION (1d EMA50) ===
        price_above_1d_ema = close[i] > ema_1d_aligned[i]
        
        # === CAMARILLA LEVELS from previous CLOSED bar (no look-ahead) ===
        prev_high = high[i - 1]
        prev_low = low[i - 1]
        prev_close = close[i - 1]
        prev_range = prev_high - prev_low
        
        if prev_range <= 1e-10:
            signals[i] = 0.0
            continue
        
        # Classic Camarilla levels
        r3 = prev_close + prev_range * 0.09167
        r4 = prev_close + prev_range * 0.18333
        s3 = prev_close - prev_range * 0.09167
        s4 = prev_close - prev_range * 0.18333
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        bars_since_entry = i - entry_bar if in_position else 999
        
        if not in_position:
            # === LONG: Price touches S3 or S4 with volume + trend alignment ===
            # Only in choppy (ranging) markets — key filter from DB winner
            if is_choppy and price_above_1d_ema and vol_spike:
                # S4 touch (deeper level = better risk/reward)
                if low[i] <= s4:
                    desired_signal = SIZE
                    last_entry_level = 1  # S4
                # S3 touch (softer level — needs stronger vol)
                elif low[i] <= s3 and vol_ratio[i] > 2.2:
                    desired_signal = SIZE
                    last_entry_level = 2  # S3
            
            # === SHORT: Price touches R3 or R4 with volume + trend alignment ===
            if is_choppy and not price_above_1d_ema and vol_spike:
                # R4 touch
                if high[i] >= r4:
                    desired_signal = -SIZE
                    last_entry_level = -1  # R4
                # R3 touch (needs stronger vol)
                elif high[i] >= r3 and vol_ratio[i] > 2.2:
                    desired_signal = -SIZE
                    last_entry_level = -2  # R3
        
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
        
        # === TAKE PROFIT: Price reverts to mid-point (mean reversion target) ===
        if in_position and bars_since_entry >= 3:  # Hold at least 3 bars (12h)
            # Mid-point of Camarilla range
            mid_point = (prev_high + prev_low) / 2
            
            # Long exit: price reverts up to mid
            if position_side > 0 and close[i] >= mid_point:
                desired_signal = 0.0
            
            # Short exit: price reverts down to mid
            if position_side < 0 and close[i] <= mid_point:
                desired_signal = 0.0
            
            # Alternative: 2:1 reward/risk take profit
            if position_side > 0:
                profit_target = entry_price + 2.0 * entry_atr
                if high[i] >= profit_target:
                    desired_signal = SIZE / 2  # Take partial profit
            if position_side < 0:
                profit_target = entry_price - 2.0 * entry_atr
                if low[i] <= profit_target:
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
                bars_since_entry = 0
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                stop_price = 0.0
                bars_since_entry = 0
        
        signals[i] = desired_signal
    
    return signals