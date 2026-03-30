#!/usr/bin/env python3
"""
Experiment #024: 12h Bollinger Band Volatility Expansion + 1d SMA Trend

HYPOTHESIS: BB lower/upper band touches during volatility expansion represent
statistically significant mean reversion points on crypto. Combined with:
- ATR ratio > 1.5 (volatility spike filters noise)
- 1d SMA200 trend (aligns with macro direction)
- Volume confirmation

WHY IT SHOULD WORK IN BOTH BULL AND BEAR:
- 2021 bull: BB lower band + htf_bullish + vol spike = strong bounce setups
- 2022 bear: BB upper band + htf_bearish + vol spike = rallies fade
- 2025 range: BB extremes work well in choppy markets

WHY 12h: Slow enough to avoid overtrading (target 80-120 total over 4 years).
Fast enough to capture 2-4 major volatility events per year per symbol.

TRADE COUNT ESTIMATE: ~1-2 entries per symbol per month = 48-96 over 4 years.
Size: 0.30.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_bb_volatility_1d_sma_v1"
timeframe = "12h"
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

def calculate_bollinger_bands(high, low, close, period=20, num_std=2.0):
    """Bollinger Bands"""
    mid = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    upper = mid + num_std * std
    lower = mid - num_std * std
    return upper, mid, lower

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d SMA200 for macro trend (call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    sma_200 = pd.Series(df_1d['close'].values).rolling(window=200, min_periods=200).mean().values
    sma_200_aligned = align_htf_to_ltf(prices, df_1d, sma_200)
    
    # === 12h Indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # ATR ratio for volatility expansion detection
    atr_30 = calculate_atr(high, low, close, period=30)
    atr_ratio = atr_14 / np.where(atr_30 > 1e-10, atr_30, 1.0)
    
    # Bollinger Bands
    bb_upper, bb_mid, bb_lower = calculate_bollinger_bands(high, low, close, period=20, num_std=2.0)
    
    # Volume ratio (precompute for speed)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 1e-10, vol_ma20, 1.0)
    
    # === Signals ===
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_bar = 0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    warmup = 100
    
    for i in range(warmup, n):
        # NaN check
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(sma_200_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Update tracking
        if in_position:
            if position_side > 0:
                highest_since_entry = max(highest_since_entry, high[i])
            else:
                lowest_since_entry = min(lowest_since_entry, low[i])
        
        # === TREND DETECTION ===
        htf_bullish = close[i] > sma_200_aligned[i]
        htf_bearish = close[i] < sma_200_aligned[i]
        
        # === VOLATILITY EXPANSION FILTER ===
        vol_expansion = atr_ratio[i] > 1.5
        
        # === VOLUME CONFIRMATION ===
        vol_confirm = vol_ratio[i] > 1.2
        
        # === BB TOUCH CONDITIONS ===
        # Lower band touch (potential bounce)
        touch_lower = close[i] <= bb_lower[i] * 1.02 if not np.isnan(bb_lower[i]) else False
        # Upper band touch (potential reversal)
        touch_upper = close[i] >= bb_upper[i] * 0.98 if not np.isnan(bb_upper[i]) else False
        
        # === MIN HOLD: 1 bar (12h) ===
        min_hold = (i - entry_bar) >= 1
        
        # === ATR TRAILING STOP ===
        def check_atr_stop():
            if not in_position:
                return False
            if position_side > 0:
                return low[i] < (highest_since_entry - 3.0 * entry_atr)
            else:
                return high[i] > (lowest_since_entry + 3.0 * entry_atr)
        
        # === EXITS ===
        if in_position:
            stop_hit = check_atr_stop()
            
            # Exit if trend reverses AND min hold met
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
            # LONG: BB lower touch + vol expansion + volume confirm + uptrend
            if touch_lower and vol_expansion and vol_confirm and htf_bullish:
                in_position = True
                position_side = 1
                entry_bar = i
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                signals[i] = SIZE
            
            # LONG: BB lower touch + strong vol expansion (even without volume)
            elif touch_lower and atr_ratio[i] > 2.0 and htf_bullish:
                in_position = True
                position_side = 1
                entry_bar = i
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                signals[i] = SIZE * 0.8
            
            # SHORT: BB upper touch + vol expansion + volume confirm + downtrend
            elif touch_upper and vol_expansion and vol_confirm and htf_bearish:
                in_position = True
                position_side = -1
                entry_bar = i
                entry_atr = atr_14[i]
                lowest_since_entry = low[i]
                signals[i] = -SIZE
            
            # SHORT: BB upper touch + strong vol expansion (even without volume)
            elif touch_upper and atr_ratio[i] > 2.0 and htf_bearish:
                in_position = True
                position_side = -1
                entry_bar = i
                entry_atr = atr_14[i]
                lowest_since_entry = low[i]
                signals[i] = -SIZE * 0.8
            
            else:
                signals[i] = 0.0
    
    return signals