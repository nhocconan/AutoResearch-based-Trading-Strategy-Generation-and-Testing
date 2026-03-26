#!/usr/bin/env python3
"""
Experiment #009: 4h Camarilla S3/R3 Single Zone + Volume Spike + Choppiness

HYPOTHESIS: Camarilla S3/R3 are key institutional order zones. Single zone 
prevents overlapping entries. Volume spike confirms smart money involvement.
Choppiness filters ranging markets. ATR stop and opposite pivot TP manage risk.

WHY THIS SHOULD WORK IN BOTH BULL AND BEAR:
- S3: Bull markets = bounce zone for longs. Bear markets = support before breakdown.
- R3: Bear markets = bounce for shorts. Bull markets = resistance before breakout.
- Choppiness ensures we only trade in trending conditions.
- Single zone (no S3+S4 overlap) = fewer trades = less fee drag.

KEY SIMPLIFICATIONS from #003:
1. Single entry zone (S3 for longs, R3 for shorts) - prevents overtrading
2. Volume spike REQUIRED - no EMA fallback (quality > quantity)
3. Removed 1d HMA trend bias - entry conditions already strict enough
4. Added 8-bar cooldown - prevents signal spam

TARGET: 60-120 total trades over 4 years (15-30/year).
DB reference: gen_camarilla_pivot_volume_spike_choppiness_4h_v1 (95tr, Sharpe=1.471)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_camarilla_single_zone_vol_chop_v1"
timeframe = "4h"
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

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures market choppiness
    CHOP > 61.8 = ranging, CHOP < 50 = trending (allow trades)
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
    Camarilla pivot levels
    S1-S4 below close, R1-R4 above close
    S3/R3 are the key reversal zones
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
        
        # S3 and S4 (support)
        pivots['s3'][i] = close - high_low_range * 1.1 / 4
        pivots['s4'][i] = close - high_low_range * 1.1 / 2
        
        # R3 and R4 (resistance)
        pivots['r3'][i] = close + high_low_range * 1.1 / 4
        pivots['r4'][i] = close + high_low_range * 1.1 / 2
    
    return pivots

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load 1d data ONCE for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Camarilla pivots
    cam_pivots = calculate_camarilla_pivots(
        df_1d['high'].values,
        df_1d['low'].values,
        df_1d['close'].values
    )
    
    # Align pivots to 4h (shifted by 1 to avoid look-ahead)
    s3_aligned = align_htf_to_ltf(prices, df_1d, cam_pivots['s3'])
    r3_aligned = align_htf_to_ltf(prices, df_1d, cam_pivots['r3'])
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    
    # Volume moving average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    bars_since_entry = 0
    
    # Cooldown tracking
    bars_since_exit = 100  # Start ready to trade
    
    # Warmup
    warmup = 60
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(chop_14[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(s3_aligned[i]) or np.isnan(r3_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(vol_ratio[i]):
            signals[i] = 0.0
            continue
        
        # === COOLDOWN CHECK ===
        bars_since_exit += 1
        bars_since_entry += 1
        
        # === REGIME CHECK ===
        chop = chop_14[i]
        is_trending = chop < 55.0  # Trending = allow entries
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === PIVOT LEVELS ===
        s3 = s3_aligned[i]
        r3 = r3_aligned[i]
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position and bars_since_exit >= 8:
            # === LONG: Price touches S3 with volume + trending ===
            if is_trending and vol_spike:
                # Price within 0.3 ATR of S3 (touched support)
                dist_to_s3 = (close[i] - s3) / atr_14[i] if atr_14[i] > 0 else 999
                if -0.3 <= dist_to_s3 <= 2.0:
                    desired_signal = SIZE
            
            # === SHORT: Price touches R3 with volume + trending ===
            if is_trending and vol_spike:
                # Price within 0.3 ATR of R3 (touched resistance)
                dist_to_r3 = (r3 - close[i]) / atr_14[i] if atr_14[i] > 0 else 999
                if -0.3 <= dist_to_r3 <= 2.0:
                    desired_signal = -SIZE
        
        # === STOPLOSS CHECK ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            # Long stop: price drops below entry - 2.5 * ATR
            stop_price = entry_price - 2.5 * entry_atr
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            # Short stop: price rises above entry + 2.5 * ATR
            stop_price = entry_price + 2.5 * entry_atr
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
            bars_since_exit = 0
        
        # === TAKE PROFIT at opposite pivot ===
        tp_triggered = False
        
        if in_position and position_side > 0:
            # TP for long: price reaches R3
            if not np.isnan(r3) and high[i] >= r3:
                tp_triggered = True
        
        if in_position and position_side < 0:
            # TP for short: price reaches S3
            if not np.isnan(s3) and low[i] <= s3:
                tp_triggered = True
        
        if tp_triggered:
            desired_signal = 0.0
            bars_since_exit = 0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                bars_since_entry = 0
        elif in_position:
            # Position closed
            in_position = False
            position_side = 0
            entry_price = 0.0
            entry_atr = 0.0
        
        signals[i] = desired_signal
    
    return signals