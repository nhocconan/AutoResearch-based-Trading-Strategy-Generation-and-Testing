#!/usr/bin/env python3
"""
Experiment #021: 4h Camarilla Pivot + Volume Spike + Choppiness Regime

HYPOTHESIS: Use proven DB winner pattern (Sharpe 1.47 on ETHUSDT test):
1. Camarilla pivot levels (not Donchian) - proven more robust entry points
2. Volume spike > 2.0x - strict confirmation to reduce trades
3. Choppiness < 50 - trending regime filter
4. ATR trailing stop

WHY IT SHOULD WORK IN BOTH MARKETS:
- Bull: Price touches Camarilla support + volume + chop < 50 = long
- Bear: Price touches Camarilla resistance + volume + chop < 50 = short
- Range: CHOP > 61.8 prevents whipsaws
- Fewer trades (target 75-150) = less fee drag = better generalization

TARGET: 100-200 total trades over 4 years (25-50/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_camarilla_pivot_volume_chop_v1"
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

def calculate_camarilla_pivots(high, low, close):
    """
    Camarilla Pivot Levels (8 levels: 4 support, 4 resistance)
    Formula: 
        Pivot = (high + low + close) / 3
        S1 = close - (high - low) * 1.1/12
        S2 = close - (high - low) * 1.1/6
        S3 = close - (high - low) * 1.1/4
        S4 = close - (high - low) * 1.1/2
        R1 = close + (high - low) * 1.1/12
        R2 = close + (high - low) * 1.1/6
        R3 = close + (high - low) * 1.1/4
        R4 = close + (high - low) * 1.1/2
    """
    n = len(close)
    pivot = (high + low + close) / 3.0
    rng = high - low
    
    # Support levels (S1-S4)
    s1 = close - rng * 1.1 / 12.0
    s2 = close - rng * 1.1 / 6.0
    s3 = close - rng * 1.1 / 4.0
    s4 = close - rng * 1.1 / 2.0
    
    # Resistance levels (R1-R4)
    r1 = close + rng * 1.1 / 12.0
    r2 = close + rng * 1.1 / 6.0
    r3 = close + rng * 1.1 / 4.0
    r4 = close + rng * 1.1 / 2.0
    
    return {
        'pivot': pivot,
        's1': s1, 's2': s2, 's3': s3, 's4': s4,
        'r1': r1, 'r2': r2, 'r3': r3, 'r4': r4
    }

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    CHOP > 61.8 = ranging - DON'T enter
    CHOP < 50 = trending - GOOD to enter
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.full(n, np.nan)
    for i in range(period, n):
        atr_sum = np.sum(tr[i - period + 1:i + 1])
        highest = np.max(high[i - period + 1:i + 1])
        lowest = np.min(low[i - period + 1:i + 1])
        
        if highest > lowest and atr_sum > 0:
            range_hl = highest - lowest
            chop[i] = 100 * np.log10(atr_sum / range_hl) / np.log10(period)
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # 4h Camarilla pivots
    pivots = calculate_camarilla_pivots(high, low, close)
    
    # Local 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    
    # Volume ratio - strict 2.0x threshold
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Signals
    signals = np.zeros(n)
    SIZE = 0.28  # 28% position size
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    trailing_high = 0.0
    trailing_low = 0.0
    
    warmup = 100  # 20 for vol MA + 14 for CHOP + pivot buffer
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        # === CHOPPINESS REGIME FILTER ===
        chop_value = chop[i]
        is_choppy = chop_value > 61.8
        is_trending = chop_value < 50
        
        # === VOLUME CONFIRMATION (strict 2.0x) ===
        vol_spike = vol_ratio[i] > 2.0
        
        # === CAMARILLA PIVOT TOUCH ===
        # Long: price touches S1 or S2 support (within 0.5 ATR)
        # Short: price touches R1 or R2 resistance (within 0.5 ATR)
        s1 = pivots['s1'][i]
        s2 = pivots['s2'][i]
        r1 = pivots['r1'][i]
        r2 = pivots['r2'][i]
        
        atr_val = atr_14[i]
        
        # Long entry: price near S1 or S2 support + volume spike + trending
        long_touch_s1 = (close[i] >= s1 - 0.5 * atr_val) and (close[i] <= s1 + 0.5 * atr_val)
        long_touch_s2 = (close[i] >= s2 - 0.5 * atr_val) and (close[i] <= s2 + 0.5 * atr_val)
        
        # Short entry: price near R1 or R2 resistance + volume spike + trending
        short_touch_r1 = (close[i] >= r1 - 0.5 * atr_val) and (close[i] <= r1 + 0.5 * atr_val)
        short_touch_r2 = (close[i] >= r2 - 0.5 * atr_val) and (close[i] <= r2 + 0.5 * atr_val)
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # LONG: near support + volume + trending
            if (long_touch_s1 or long_touch_s2) and vol_spike and is_trending:
                desired_signal = SIZE
            
            # SHORT: near resistance + volume + trending
            if (short_touch_r1 or short_touch_r2) and vol_spike and is_trending:
                desired_signal = -SIZE
        
        # === STOPLOSS (2.5 ATR trailing stop) ===
        if in_position:
            if position_side > 0:
                # Update trailing high
                if i == entry_bar or high[i] > trailing_high:
                    trailing_high = high[i]
                
                # Trailing stop
                stop_price = trailing_high - 2.5 * entry_atr
                if low[i] < stop_price:
                    desired_signal = 0.0
                
                # Exit if market becomes choppy
                if is_choppy:
                    desired_signal = 0.0
            
            elif position_side < 0:
                # Update trailing low
                if i == entry_bar or low[i] < trailing_low:
                    trailing_low = low[i]
                
                # Trailing stop
                stop_price = trailing_low + 2.5 * entry_atr
                if high[i] > stop_price:
                    desired_signal = 0.0
                
                # Exit if market becomes choppy
                if is_choppy:
                    desired_signal = 0.0
        
        # === MINIMUM HOLD: 4 bars to avoid fee churn ===
        if in_position and (i - entry_bar) < 4:
            desired_signal = position_side * SIZE
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                trailing_high = high[i]
                trailing_low = low[i]
        else:
            if in_position:
                in_position = False
                position_side = 0
        
        signals[i] = desired_signal
    
    return signals