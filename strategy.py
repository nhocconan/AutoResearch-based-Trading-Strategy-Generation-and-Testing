#!/usr/bin/env python3
"""
Experiment #021: 12h Camarilla Pivot + Volume Spike + 1d Trend

HYPOTHESIS: Camarilla pivots are institutional support/resistance levels that
work in both bull and bear markets. Price bouncing from S1/S2 with volume 
confirms smart money accumulation. Price rejecting at R1/R2 with volume 
confirms distribution. Combined with 1d HMA trend filter to avoid fighting 
the larger trend. 12h is slow enough to avoid overtrading but fast enough 
to capture meaningful moves.

WHY BOTH MARKETS:
- Bull: longs at S1/S2 bounces when price above 1d HMA
- Bear: shorts at R1/R2 rejections when price below 1d HMA
- Range: mean-revert to center pivot

TIMEFRAME: 12h primary | HTF: 1d
TARGET: 50-150 total trades over 4 years (12-37/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_camarilla_vol_1d_v1"
timeframe = "12h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = max(1, period // 2)
    sqrt_n = max(1, int(np.sqrt(period)))
    
    def wma(series, span):
        result = np.full(len(series), np.nan, dtype=np.float64)
        weights = np.arange(1, span + 1, dtype=np.float64)
        weight_sum = np.sum(weights)
        for i in range(span - 1, len(series)):
            if not np.isnan(series[i]):
                window = series[i - span + 1:i + 1].astype(np.float64)
                if not np.any(np.isnan(window)):
                    result[i] = np.sum(window * weights) / weight_sum
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff[i] = 2.0 * wma_half[i] - wma_full[i]
    
    return wma(diff, sqrt_n)

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

def calculate_camarilla_levels(high, low, close, period=20):
    """
    Classic Camarilla pivot levels.
    Based on market structure of last N bars.
    """
    n = len(close)
    
    # Pivot point
    pivot = (high + low + close) / 3.0
    
    # Range
    rng = high - low
    
    # Resistance levels
    r4 = close + rng * 0.55
    r3 = close + rng * 0.275
    r2 = close + rng * 0.183
    r1 = close + rng * 0.0916
    
    # Support levels
    s1 = close - rng * 0.0916
    s2 = close - rng * 0.183
    s3 = close - rng * 0.275
    s4 = close - rng * 0.55
    
    return {
        'pivot': pivot,
        'r1': r1, 'r2': r2, 'r3': r3, 'r4': r4,
        's1': s1, 's2': s2, 's3': s3, 's4': s4
    }

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d HMA for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate local 12h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Camarilla levels (20-period lookback for structure)
    levels = calculate_camarilla_levels(high, low, close, period=20)
    s1 = levels['s1']
    s2 = levels['s2']
    r1 = levels['r1']
    r2 = levels['r2']
    
    # Volume MA for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # ATR-based proximity threshold (0.5% of price OR 0.3 ATR)
    atr_threshold = 0.3 * atr_14
    pct_threshold = close * 0.005
    threshold = np.maximum(atr_threshold, pct_threshold)
    
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = 50
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(s1[i]) or np.isnan(r1[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === TREND FILTER (1d HMA) ===
        bullish_trend = close[i] > hma_1d_aligned[i]
        bearish_trend = close[i] < hma_1d_aligned[i]
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === PROXIMITY TO PIVOT LEVELS ===
        # Distance from current price to key levels
        dist_to_s1 = abs(close[i] - s1[i])
        dist_to_s2 = abs(close[i] - s2[i])
        dist_to_r1 = abs(close[i] - r1[i])
        dist_to_r2 = abs(close[i] - r2[i])
        
        # Near level if within threshold
        near_s1 = dist_to_s1 < threshold[i]
        near_s2 = dist_to_s2 < threshold[i]
        near_r1 = dist_to_r1 < threshold[i]
        near_r2 = dist_to_r2 < threshold[i]
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG ENTRY: Price near S1/S2 + volume spike + bullish trend ===
            # Support bounce: price above level but touching
            s1_touch = low[i] <= s1[i] + threshold[i] and close[i] > s1[i]
            s2_touch = low[i] <= s2[i] + threshold[i] and close[i] > s2[i]
            
            if (s1_touch or s2_touch) and vol_spike and bullish_trend:
                desired_signal = SIZE
            
            # === SHORT ENTRY: Price near R1/R2 + volume spike + bearish trend ===
            # Resistance rejection: price below level but touching
            r1_touch = high[i] >= r1[i] - threshold[i] and close[i] < r1[i]
            r2_touch = high[i] >= r2[i] - threshold[i] and close[i] < r2[i]
            
            if (r1_touch or r2_touch) and vol_spike and bearish_trend:
                desired_signal = -SIZE
        
        # === STOPLOSS CHECK (2.5 ATR) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            if low[i] < trailing_stop:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            if high[i] > trailing_stop:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === TAKE PROFIT: When price reaches opposite level ===
        tp_triggered = False
        
        if in_position and position_side > 0:
            # TP: price reaches R1 or R2
            if close[i] >= r1[i] * 0.998:
                tp_triggered = True
            if close[i] >= r2[i] * 0.998:
                tp_triggered = True
        
        if in_position and position_side < 0:
            # TP: price reaches S1 or S2
            if close[i] <= s1[i] * 1.002:
                tp_triggered = True
            if close[i] <= s2[i] * 1.002:
                tp_triggered = True
        
        if tp_triggered:
            desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New position or flip
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
            # else: maintain position (no churn)
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = desired_signal
    
    return signals