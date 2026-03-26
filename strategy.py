#!/usr/bin/env python3
"""
Experiment #021: 4h Camarilla Pullback + Tight Zone + Volume + 1d Trend

HYPOTHESIS: Camarilla S3/S4 zones work for longs when price pulls back TO the level
(within ±0.5 ATR, NOT up to 2 ATR). This catches mean-reversion entries at institutional
support zones. The #016 failure (2443 trades) had a 0-2 ATR zone that was far too loose.

KEY DIFFERENCE from #016:
- Tight zone: ±0.5 ATR (not 0-2 ATR) → ~4x fewer entries
- AND logic for trend+volume (not OR) → higher conviction
- Choppiness < 50 (not <61.8) → trending market required
- Minimum 4-bar hold to prevent chop whipsaw

TARGET: 75-150 total trades over 4 years (~19-37/year)
DB reference: gen_camarilla_pivot_volume_spike_choppiness_4h_v1 (95 trades, Sharpe=1.471)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_camarilla_pullback_tight_v1"
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
    Choppiness Index
    CHOP > 61.8 = ranging, CHOP < 38.2 = trending
    We require CHOP < 50 for entry (more selective than <61.8)
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
    Camarilla pivot levels (from previous day's HLC)
    S3/S4 = support zones for longs (price should be AT or slightly BELOW)
    R3/R4 = resistance zones for shorts (price should be AT or slightly ABOVE)
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
        
        # S3/S4 = support (below close)
        pivots['s3'][i] = close - high_low_range * 1.1 / 4
        pivots['s4'][i] = close - high_low_range * 1.1 / 2
        # R3/R4 = resistance (above close)
        pivots['r3'][i] = close + high_low_range * 1.1 / 4
        pivots['r4'][i] = close + high_low_range * 1.1 / 2
    
    return pivots

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HMA for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate Camarilla pivots from 1d
    cam_pivots = calculate_camarilla_pivots(
        df_1d['high'].values,
        df_1d['low'].values,
        df_1d['close'].values
    )
    
    # Align pivots to 4h (with shift(1) for completed bars)
    s3_aligned = align_htf_to_ltf(prices, df_1d, cam_pivots['s3'])
    s4_aligned = align_htf_to_ltf(prices, df_1d, cam_pivots['s4'])
    r3_aligned = align_htf_to_ltf(prices, df_1d, cam_pivots['r3'])
    r4_aligned = align_htf_to_ltf(prices, df_1d, cam_pivots['r4'])
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    
    # Volume MA and ratio
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
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
    bars_in_trade = 0
    
    warmup = 50
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(chop_14[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(s3_aligned[i]) or np.isnan(r3_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        bars_in_trade += 1
        
        # === REGIME CHECK (stricter: CHOP < 50) ===
        chop = chop_14[i]
        is_trending = chop < 50.0  # Stricter than <61.8
        
        # === TREND BIAS (1d HMA) ===
        price_above_1d_hma = close[i] > hma_1d_aligned[i] if not np.isnan(hma_1d_aligned[i]) else True
        
        # === VOLUME CONFIRMATION (1.5x, stricter) ===
        vol_spike = vol_ratio[i] > 1.5  # Stricter than 1.3
        
        # === CAMARILLA PIVOT LEVELS ===
        s3 = s3_aligned[i]
        s4 = s4_aligned[i]
        r3 = r3_aligned[i]
        r4 = r4_aligned[i]
        
        # TIGHT ZONE: price must be WITHIN ±0.5 ATR of pivot (not 0-2 ATR!)
        # For LONG: price at or below S3/S4 support (within 0.5 ATR below)
        # For SHORT: price at or above R3/R4 resistance (within 0.5 ATR above)
        at_s3_zone = False
        at_s4_zone = False
        at_r3_zone = False
        at_r4_zone = False
        
        ZONE_SIZE = 0.5  # Tighter zone than original 2.0
        
        if not np.isnan(s3) and atr_14[i] > 0:
            # dist < 0 means price at or below S3
            # dist between -0.5 and 0 means price within 0.5 ATR of S3
            dist_s3 = (close[i] - s3) / atr_14[i]
            at_s3_zone = (-ZONE_SIZE <= dist_s3 <= 0.0)
        
        if not np.isnan(s4) and atr_14[i] > 0:
            dist_s4 = (close[i] - s4) / atr_14[i]
            at_s4_zone = (-ZONE_SIZE <= dist_s4 <= 0.0)
        
        if not np.isnan(r3) and atr_14[i] > 0:
            # dist < 0 means price at or below R3 (NOT at resistance)
            # For SHORT: need price at or above R3
            dist_r3 = (close[i] - r3) / atr_14[i]
            at_r3_zone = (0.0 <= dist_r3 <= ZONE_SIZE)
        
        if not np.isnan(r4) and atr_14[i] > 0:
            dist_r4 = (close[i] - r4) / atr_14[i]
            at_r4_zone = (0.0 <= dist_r4 <= ZONE_SIZE)
        
        # === ENTRY LOGIC (AND logic: both trend AND volume required) ===
        desired_signal = 0.0
        
        if is_trending:
            # LONG: At S3/S4 support zone AND bullish 1d trend AND volume spike
            if (at_s3_zone or at_s4_zone):
                if price_above_1d_hma and vol_spike:
                    desired_signal = SIZE
            
            # SHORT: At R3/R4 resistance zone AND bearish 1d trend AND volume spike
            if (at_r3_zone or at_r4_zone):
                if (not price_above_1d_hma) and vol_spike:
                    desired_signal = -SIZE
        
        # === MINIMUM HOLDING PERIOD (prevent chop whipsaw) ===
        MIN_BARS = 4
        if bars_in_trade < MIN_BARS:
            # Keep position even if signals say exit
            if in_position:
                signals[i] = position_side * SIZE
                continue
        
        # === STOPLOSS CHECK (2.5 ATR) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === SIGNAL FLIP LOGIC ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New position or flip
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                bars_in_trade = 0
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
            # If same side, keep position (no signal churn)
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
                bars_in_trade = 0
        
        signals[i] = desired_signal if desired_signal != 0.0 else (position_side * SIZE if in_position else 0.0)
    
    return signals