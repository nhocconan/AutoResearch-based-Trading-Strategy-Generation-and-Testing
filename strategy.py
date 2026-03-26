#!/usr/bin/env python3
"""
Experiment #003 v3: Simplified 4h Camarilla + Volume + Choppiness

HYPOTHESIS: Camarilla S3/R3 levels from 1d are key support/resistance where 
institutions place orders. Volume spike confirms institutional involvement.
Choppiness Index filters ranging markets.

WHY THIS SHOULD WORK IN BOTH BULL AND BEAR:
- Camarilla works in ALL markets (derived from previous day's range)
- Bear: short at R3/R4. Bull: long at S3/S4. Range: mean-revert between pivots
- Simple entry: price within 2 ATR of S3/R3 + (volume spike OR trending chop)

TARGET: 75-150 trades over 4 years (proven pattern from DB).
KEY FIX: Remove nested conditions that caused 0 trades in v1.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_camarilla_vol_chop_simple_v3"
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
    Choppiness Index - measures market choppiness
    CHOP > 61.8 = ranging (avoid), CHOP < 50 = trending (trade)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i - period + 1:i + 1])
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and atr_sum > 0:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop

def calculate_camarilla_pivots(prev_high, prev_low, prev_close):
    """
    Camarilla pivot levels from previous day's HLC
    S3 = close - range * 1.1/4
    R3 = close + range * 1.1/4
    S4 = close - range * 1.1/2
    R4 = close + range * 1.1/2
    """
    n = len(prev_high)
    pivots = {
        's3': np.full(n, np.nan, dtype=np.float64),
        's4': np.full(n, np.nan, dtype=np.float64),
        'r3': np.full(n, np.nan, dtype=np.float64),
        'r4': np.full(n, np.nan, dtype=np.float64),
    }
    
    for i in range(n):
        if np.isnan(prev_high[i]) or np.isnan(prev_low[i]) or np.isnan(prev_close[i]):
            continue
        
        high_low_range = prev_high[i] - prev_low[i]
        if high_low_range <= 1e-10:
            continue
        
        close = prev_close[i]
        
        pivots['s3'][i] = close - high_low_range * 1.1 / 4
        pivots['s4'][i] = close - high_low_range * 1.1 / 2
        pivots['r3'][i] = close + high_low_range * 1.1 / 4
        pivots['r4'][i] = close + high_low_range * 1.1 / 2
    
    return pivots

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load 1d data for Camarilla pivots (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Camarilla pivots
    cam_pivots = calculate_camarilla_pivots(
        df_1d['high'].values,
        df_1d['low'].values,
        df_1d['close'].values
    )
    
    # Align pivots to 4h (auto shift by 1 HTF bar for no look-ahead)
    s3_aligned = align_htf_to_ltf(prices, df_1d, cam_pivots['s3'])
    s4_aligned = align_htf_to_ltf(prices, df_1d, cam_pivots['s4'])
    r3_aligned = align_htf_to_ltf(prices, df_1d, cam_pivots['r3'])
    r4_aligned = align_htf_to_ltf(prices, df_1d, cam_pivots['r4'])
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    
    # Volume moving average (20-period for 4h = ~80h of volume)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    stop_price = 0.0
    
    # Warmup (need 100 bars for indicators)
    warmup = 100
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(chop_14[i]) or np.isnan(vol_ratio[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(s3_aligned[i]) or np.isnan(r3_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # Get pivot values
        s3 = s3_aligned[i]
        s4 = s4_aligned[i]
        r3 = r3_aligned[i]
        r4 = r4_aligned[i]
        
        # Calculate distances to pivots (as ATR multiples)
        if atr_14[i] > 0:
            dist_to_s3 = (close[i] - s3) / atr_14[i]
            dist_to_s4 = (close[i] - s4) / atr_14[i] if not np.isnan(s4) else 999
            dist_to_r3 = (r3 - close[i]) / atr_14[i]
            dist_to_r4 = (r4 - close[i]) / atr_14[i] if not np.isnan(r4) else 999
        else:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === FILTERS ===
        # Volume spike: >1.5x average volume
        vol_spike = vol_ratio[i] > 1.5
        # Choppiness trending: <55 (lower = more trending)
        is_trending = chop_14[i] < 55.0
        
        # Need EITHER volume spike OR trending (relaxed condition)
        # This ensures we still get trades even without volume spike
        tradeable = vol_spike or is_trending
        
        # === STOPLOSS CHECK ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.0 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.0 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            in_position = False
            position_side = 0
            signals[i] = 0.0
            continue
        
        # === ENTRY LOGIC (simplified - no nested conditions) ===
        desired_signal = 0.0
        
        # Only enter if not in position
        if not in_position and tradeable:
            # LONG: price within 2 ATR of S3 or S4 support
            if dist_to_s3 >= -0.5 and dist_to_s3 < 2.5:
                desired_signal = SIZE
            elif dist_to_s4 >= -0.5 and dist_to_s4 < 2.5:
                desired_signal = SIZE
            
            # SHORT: price within 2 ATR of R3 or R4 resistance
            if dist_to_r3 >= -0.5 and dist_to_r3 < 2.5:
                desired_signal = -SIZE
            elif dist_to_r4 >= -0.5 and dist_to_r4 < 2.5:
                desired_signal = -SIZE
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            # Enter new position or reverse
            in_position = True
            position_side = int(np.sign(desired_signal))
            entry_atr = atr_14[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            if position_side > 0:
                stop_price = close[i] - 2.0 * entry_atr
            else:
                stop_price = close[i] + 2.0 * entry_atr
        
        # Hold position until stopped out
        if in_position:
            desired_signal = SIZE * position_side
        
        signals[i] = desired_signal
    
    return signals