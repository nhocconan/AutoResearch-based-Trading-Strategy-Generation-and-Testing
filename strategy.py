#!/usr/bin/env python3
"""
Experiment #022: 1d Camarilla Pivot + Williams %R + 1w Trend

HYPOTHESIS: Daily Camarilla pivot levels (S3/S4/R3/R4) mark institutional 
reversal points. Combined with Williams %R for momentum and 1w HMA for 
trend, this captures mean-reversion moves at major levels.
Simple = fewer trades = less fee drag.

TIMEFRAME: 1d primary
HTF: 1w for trend confirmation
TARGET: 50-100 total trades over 4 years (12-25/year)
ENTRY: 2 conditions (Camarilla touch + Williams %R extreme)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_camarilla_williams_1w_v1"
timeframe = "1d"
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

def calculate_williams_r(high, low, close, period=14):
    """Williams %R - momentum indicator"""
    n = len(close)
    willr = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        period_high = np.max(high[i - period + 1:i + 1])
        period_low = np.min(low[i - period + 1:i + 1])
        if period_high != period_low:
            willr[i] = -100 * (period_high - close[i]) / (period_high - period_low)
    
    return willr

def calculate_camarilla_pivots(high, low, close, period=20):
    """Camarilla pivot levels - S3, S4, R3, R4"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    s3 = np.full(n, np.nan, dtype=np.float64)
    s4 = np.full(n, np.nan, dtype=np.float64)
    r3 = np.full(n, np.nan, dtype=np.float64)
    r4 = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        h = high[i - period + 1:i + 1]
        l = low[i - period + 1:i + 1]
        pivot = (h.max() + l.min() + close[i - 1]) / 3.0
        range_val = h.max() - l.min()
        
        s3[i] = close[i - 1] + range_val * 1.1 / 4.0
        s4[i] = close[i - 1] + range_val * 1.1 / 2.0
        r3[i] = close[i - 1] - range_val * 1.1 / 4.0
        r4[i] = close[i - 1] - range_val * 1.1 / 2.0
    
    return s3, s4, r3, r4

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE ===
    df_1w = get_htf_data(prices, '1w')
    
    # 1w HMA for trend
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    willr = calculate_williams_r(high, low, close, period=14)
    s3, s4, r3, r4 = calculate_camarilla_pivots(high, low, close, period=20)
    
    # Volume MA for confirmation
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
        
        if np.isnan(willr[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === 1w TREND ===
        price_above_1w_hma = close[i] > hma_1w_aligned[i]
        bull_trend = price_above_1w_hma
        bear_trend = not price_above_1w_hma
        
        # === WILLIAMS %R MOMENTUM ===
        willr_val = willr[i]
        oversold = willr_val < -80
        overbought = willr_val > -20
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === CAMARILLA LEVEL TOUCH ===
        # Long: price touches S3 or S4 (support bounce)
        touch_s3 = low[i] <= s3[i] if not np.isnan(s3[i]) else False
        touch_s4 = low[i] <= s4[i] if not np.isnan(s4[i]) else False
        
        # Short: price touches R3 or R4 (resistance rejection)
        touch_r3 = high[i] >= r3[i] if not np.isnan(r3[i]) else False
        touch_r4 = high[i] >= r4[i] if not np.isnan(r4[i]) else False
        
        # === ENTRY LOGIC (2 conditions) ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG ENTRY: Support touch + oversold + trend aligned ===
            # Condition 1: Price touches S3 or S4
            # Condition 2: Williams %R oversold OR volume spike
            if (touch_s3 or touch_s4) and (oversold or vol_spike):
                # Only in bull trend or neutral
                if bull_trend or not bear_trend:
                    desired_signal = SIZE
            
            # === SHORT ENTRY: Resistance touch + overbought ===
            if (touch_r3 or touch_r4) and (overbought or vol_spike):
                # Only in bear trend or neutral
                if bear_trend or not bull_trend:
                    desired_signal = -SIZE
        
        # === STOPLOSS (2.5 ATR) ===
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
        
        # === EXIT ===
        exit_triggered = False
        
        if in_position and position_side > 0:
            # Long exit: Williams %R overbought OR price at R3/R4
            if willr_val > -20:
                exit_triggered = True
            if touch_r3 or touch_r4:
                exit_triggered = True
        
        if in_position and position_side < 0:
            # Short exit: Williams %R oversold OR price at S3/S4
            if willr_val < -80:
                exit_triggered = True
            if touch_s3 or touch_s4:
                exit_triggered = True
        
        if exit_triggered:
            desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
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
            else:
                bars_in_trade += 1
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
        
        signals[i] = desired_signal
    
    return signals