#!/usr/bin/env python3
"""
Experiment #023: 6h Camarilla Pivot Mean Reversion + 1d SMA + Volume

HYPOTHESIS: Camarilla S3/R3 levels are statistically significant reversal
points. Price tends to bounce from these extremes back toward the mean.
Combined with 1d SMA trend filter and volume confirmation, this should:
- Work in 2021 bull: fade deep dips to S3 in uptrend
- Work in 2022 bear: rally fades to R3 in downtrend  
- Work in 2025 range: mean reversion between S3/R3

KEY INSIGHT: Previous Donchian strategies chase breakouts. This fades
extremes at known pivot levels - opposite approach, different edge.

TRADE COUNT: 75-150 total over 4 years (18-37/year).
Size: 0.30.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_camarilla_1d_sma_vol_v1"
timeframe = "6h"
leverage = 1.0

def calculate_camarilla(high, low, close, open_price=None):
    """
    Camarilla pivot levels.
    R3 = close + (high - low) * 1.1
    R4 = close + (high - low) * 1.2
    R2 = close + (high - low) * 1.1 / 2
    S3 = close - (high - low) * 1.1
    S4 = close - (high - low) * 1.2
    S2 = close - (high - low) * 1.1 / 2
    Pivot = (high + low + close) / 3
    """
    n = len(close)
    rng = high - low
    
    r4 = close + rng * 1.2
    r3 = close + rng * 1.1
    r2 = close + rng * 0.55
    pivot = (high + low + close) / 3.0
    s2 = close - rng * 0.55
    s3 = close - rng * 1.1
    s4 = close - rng * 1.2
    
    return r4, r3, r2, pivot, s2, s3, s4

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

def calculate_donchian(high, low, period=20):
    """Donchian channel for trend direction"""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d SMA for macro trend (call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    sma_1d_50 = pd.Series(df_1d['close'].values).rolling(window=50, min_periods=50).mean().values
    sma_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_1d_50)
    
    # === 6h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    r4, r3, r2, pivot, s2, s3, s4 = calculate_camarilla(high, low, close)
    
    # Donchian 20 for local trend
    dc_upper_20, dc_lower_20 = calculate_donchian(high, low, period=20)
    
    # Volume spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 1e-10, vol_ma, 1.0)
    
    # === Signals ===
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    warmup = 60
    
    for i in range(warmup, n):
        # NaN check
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(sma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Update highest/lowest for trailing stop
        if in_position:
            if position_side > 0:
                highest_since_entry = max(highest_since_entry, high[i])
            else:
                lowest_since_entry = min(lowest_since_entry, low[i])
        
        # === TREND DETECTION ===
        # 1d SMA for macro direction
        htf_bullish = close[i] > sma_1d_aligned[i]
        htf_bearish = close[i] < sma_1d_aligned[i]
        
        # Local Donchian for medium trend
        local_bullish = close[i] > dc_upper_20[i] if not np.isnan(dc_upper_20[i]) else False
        local_bearish = close[i] < dc_lower_20[i] if not np.isnan(dc_lower_20[i]) else False
        
        # === CAMARILLA SIGNALS ===
        # Price approaching S3 from above = potential long bounce
        near_s3 = (close[i] < s3[i] * 1.02) and (close[i] > s3[i] * 0.97)
        # Price approaching R3 from below = potential short bounce
        near_r3 = (close[i] > r3[i] * 0.98) and (close[i] < r3[i] * 1.03)
        
        # Price extreme: near S4 (very oversold) or R4 (very overbought)
        near_s4 = close[i] < s4[i] * 1.03
        near_r4 = close[i] > r4[i] * 0.97
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === MINIMUM HOLD: 2 bars (12h) ===
        min_hold = (i - entry_bar) >= 2
        
        # === ATR TRAILING STOP (2.5x ATR) ===
        def check_atr_stop():
            if not in_position:
                return False
            if position_side > 0:
                return low[i] < (highest_since_entry - 2.5 * entry_atr)
            else:
                return high[i] > (lowest_since_entry + 2.5 * entry_atr)
        
        # === EXITS ===
        if in_position:
            stop_hit = check_atr_stop()
            
            # Opposite trend signal exits
            if position_side > 0 and htf_bearish and min_hold:
                stop_hit = True
            if position_side < 0 and htf_bullish and min_hold:
                stop_hit = True
            
            if stop_hit:
                signals[i] = 0.0
                in_position = False
                position_side = 0
            else:
                signals[i] = position_side * SIZE
        
        # === NEW POSITIONS ===
        if not in_position:
            # LONG: Price at/below S3 + volume spike + 1d uptrend
            if near_s3 and vol_spike and htf_bullish:
                in_position = True
                position_side = 1
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                highest_since_entry = high[i]
                signals[i] = SIZE
            
            # LONG EXTRA: At S4 extreme with any volume in uptrend
            elif near_s4 and htf_bullish:
                in_position = True
                position_side = 1
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                highest_since_entry = high[i]
                signals[i] = SIZE * 0.5  # Half size for S4 (more risky)
            
            # SHORT: Price at/above R3 + volume spike + 1d downtrend
            elif near_r3 and vol_spike and htf_bearish:
                in_position = True
                position_side = -1
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                lowest_since_entry = low[i]
                signals[i] = -SIZE
            
            # SHORT EXTRA: At R4 extreme with any volume in downtrend
            elif near_r4 and htf_bearish:
                in_position = True
                position_side = -1
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                lowest_since_entry = low[i]
                signals[i] = -SIZE * 0.5  # Half size for R4
            
            else:
                signals[i] = 0.0
    
    return signals