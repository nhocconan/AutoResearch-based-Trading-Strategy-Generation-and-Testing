#!/usr/bin/env python3
"""
Experiment #011: 6h Williams %R + Weekly Pivot Mean Reversion

HYPOTHESIS: Markets oscillate around institutional price levels. When price reaches
weekly pivot S3 (support) or R3 (resistance) AND Williams %R confirms oversold/overbought
extremes, price mean-reverts to the pivot level. Combined with 1d EMA50 trend filter
to avoid fighting the trend, this catches reversals at key levels.

WHY IT WORKS IN BULL AND BEAR: Pivot levels are symmetrical — S3 acts as support
in both bull markets AND bear rallies. R3 acts as resistance in both. Williams %R
identifies when price has extended too far from fair value.

DIFFERENTIATION: Williams %R is momentum-based, not RSI-based (all recent failures
used RSI variants). Weekly pivots on 6h capture the same structure as 12h but with
more entry granularity.

TARGET: 75-150 total trades over 4 years = 19-37/year.
Signal size: 0.25.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_willr_pivot_meanrev_1d_v1"
timeframe = "6h"
leverage = 1.0

def calculate_williams_r(high, low, close, period=14):
    """Williams %R - momentum indicator"""
    n = len(close)
    willr = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        
        if highest_high != lowest_low:
            willr[i] = -100 * (highest_high - close[i]) / (highest_high - lowest_low)
    
    return willr

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

def calculate_pivot_levels(high, low, close, lookback=1):
    """Classic Pivot Points with support/resistance levels"""
    n = len(close)
    pivots = np.zeros((n, 5))  # pivot, r1, s1, r2, s2
    
    for i in range(lookback, n):
        prev_high = high[i - lookback]
        prev_low = low[i - lookback]
        prev_close = close[i - lookback]
        
        # Classic pivot calculation
        pivot = (prev_high + prev_low + prev_close) / 3.0
        
        # Support levels
        s1 = 2 * pivot - prev_high
        s2 = pivot - (prev_high - prev_low)
        s3 = prev_low - 2 * (prev_high - pivot)
        
        # Resistance levels
        r1 = 2 * pivot - prev_low
        r2 = pivot + (prev_high - prev_low)
        r3 = prev_high + 2 * (pivot - prev_low)
        
        pivots[i] = [pivot, r1, r1, s1, s1]  # Using R1/S1 for tighter bands
        pivots[i] = [pivot, r3, r2, s3, s2]  # Full range
    
    return pivots

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
    
    # === Local 6h indicators ===
    willr_14 = calculate_williams_r(high, low, close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Volume ratio
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # === Weekly pivot levels (1d data = weekly on 6h) ===
    # Pivot points from 1d HTF
    pivot_1d = np.zeros(n)
    r1_1d = np.zeros(n)
    r3_1d = np.zeros(n)
    s1_1d = np.zeros(n)
    s3_1d = np.zeros(n)
    
    # Calculate pivots from 1d data aligned to 6h
    for i in range(1, len(df_1d)):
        prev_h = df_1d['high'].values[i - 1]
        prev_l = df_1d['low'].values[i - 1]
        prev_c = df_1d['close'].values[i - 1]
        
        piv = (prev_h + prev_l + prev_c) / 3.0
        r3_val = prev_h + 2 * (piv - prev_l)
        r1_val = 2 * piv - prev_l
        s1_val = 2 * piv - prev_h
        s3_val = prev_l - 2 * (prev_h - piv)
        
        # Find corresponding 6h bars for this 1d bar
        day_time = df_1d['open_time'].values[i - 1]
        for j in range(i * 4, min((i + 1) * 4, n)):
            if j > 0:
                pivot_1d[j] = piv
                r1_1d[j] = r1_val
                r3_1d[j] = r3_val
                s1_1d[j] = s1_val
                s3_1d[j] = s3_val
    
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
    
    warmup = 150  # Need enough for alignment buffer
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(willr_14[i]) or np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # Skip if EMA not aligned
        if np.isnan(ema_1d_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # Skip if pivot not calculated
        if pivot_1d[i] == 0:
            signals[i] = 0.0
            continue
        
        # === TREND DIRECTION (1d EMA50) ===
        price_above_1d_ema = close[i] > ema_1d_aligned[i]
        
        # === WILLIAMS %R CONFIRMATION ===
        # -100 to 0 scale: < -80 = oversold, > -20 = overbought
        willr_oversold = willr_14[i] < -80
        willr_overbought = willr_14[i] > -20
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === PIVOT LEVELS ===
        r3 = r3_1d[i]
        s3 = s3_1d[i]
        
        # Distance to pivot levels as % of price
        dist_to_r3 = (high[i] - r3) / close[i] if r3 > 0 else 999
        dist_to_s3 = (s3 - low[i]) / close[i] if s3 > 0 else 999
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG: Price approaches S3 + oversold + volume + uptrend ===
            if price_above_1d_ema and willr_oversold and vol_spike:
                # Price within 0.5% of S3
                if dist_to_s3 < 0.005:
                    desired_signal = SIZE
                # Price has bounced from below S3
                elif low[i] < s3 and close[i] > s3:
                    desired_signal = SIZE
            
            # === SHORT: Price approaches R3 + overbought + volume + downtrend ===
            if not price_above_1d_ema and willr_overbought and vol_spike:
                # Price within 0.5% of R3
                if dist_to_r3 < 0.005:
                    desired_signal = -SIZE
                # Price has rejected from above R3
                elif high[i] > r3 and close[i] < r3:
                    desired_signal = -SIZE
        
        # === STOPLOSS (2.5 ATR from entry) ===
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
        
        # === HOLD PERIOD (minimum 2 bars = 12h to avoid churn) ===
        bars_held = i - entry_bar
        
        if in_position and bars_held >= 2:
            # Take profit when Williams %R reverts to neutral
            if position_side > 0 and willr_14[i] > -40:
                desired_signal = 0.0
            if position_side < 0 and willr_14[i] < -60:
                desired_signal = 0.0
        
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