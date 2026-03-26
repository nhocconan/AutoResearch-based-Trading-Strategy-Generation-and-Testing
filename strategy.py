#!/usr/bin/env python3
"""
Experiment #008: 12h Camarilla S3/R3 + 1w Trend + Volume Spike

HYPOTHESIS: Weekly trend direction (SMA50 on 1w) provides market bias.
12h price touching Camarilla S3/R3 levels from 1d captures institutional zones.
Volume spike confirms order flow. Simple = fewer trades = less fee drag.

WHY THIS SHOULD WORK IN BOTH BULL AND BEAR:
- 1w SMA50 trend filter adapts to ANY market regime (bull, bear, range)
- Bull: only long S3 touches when 1w uptrend
- Bear: only short R3 touches when 1w downtrend
- Range: S3/R3 touches are mean-reversion trades with tight stops

DB REFERENCE: gen_camarilla_pivot_volume_spike_choppiness_4h_v1 (Sharpe=1.471)
TARGET: 75-125 total trades over 4 years (19-31/year on 12h)

KEY DESIGN (minimal conditions):
1. 1w SMA50 for trend direction (simple, robust)
2. 1d Camarilla S3/R3 as entry zones
3. Volume spike >1.5x 20-avg confirmation
4. ATR stoploss (2x) and take profit at opposite pivot
5. NO choppiness filter (causes 0 trades in backtests)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_camarilla_s3r3_1w_trend_v1"
timeframe = "12h"
leverage = 1.0

def calculate_sma(close, period):
    """Simple Moving Average with min_periods"""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

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

def calculate_camarilla_pivots(prev_high, prev_low, prev_close):
    """
    Camarilla pivot levels (classic formula)
    S3 = close - (high - low) * 1.1 / 4
    R3 = close + (high - low) * 1.1 / 4
    """
    n = len(prev_high)
    pivots = {
        's3': np.full(n, np.nan, dtype=np.float64),
        'r3': np.full(n, np.nan, dtype=np.float64),
    }
    
    for i in range(n):
        if np.isnan(prev_high[i]) or np.isnan(prev_low[i]) or np.isnan(prev_close[i]):
            continue
        
        high_low_range = prev_high[i] - prev_low[i]
        if high_low_range <= 1e-10:
            continue
        
        close = prev_close[i]
        pivots['s3'][i] = close - high_low_range * 1.1 / 4
        pivots['r3'][i] = close + high_low_range * 1.1 / 4
    
    return pivots

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load 1w data for trend direction (ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    
    # 1w SMA50 for trend direction
    sma_1w_raw = calculate_sma(df_1w['close'].values, period=50)
    sma_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_1w_raw)
    
    # Load 1d data for Camarilla pivots (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # 1d Camarilla S3/R3
    cam_pivots = calculate_camarilla_pivots(
        df_1d['high'].values,
        df_1d['low'].values,
        df_1d['close'].values
    )
    
    # Align pivots to 12h
    s3_aligned = align_htf_to_ltf(prices, df_1d, cam_pivots['s3'])
    r3_aligned = align_htf_to_ltf(prices, df_1d, cam_pivots['r3'])
    
    # 12h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Volume moving average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup (need 1w SMA50 = ~50 weeks of 1w data)
    warmup = 200
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
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
        
        if np.isnan(vol_ratio[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === 1w TREND DIRECTION ===
        price_above_1w_sma = close[i] > sma_1w_aligned[i] if not np.isnan(sma_1w_aligned[i]) else True
        is_bullish_1w = price_above_1w_sma
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === PIVOT LEVELS ===
        s3 = s3_aligned[i]
        r3 = r3_aligned[i]
        
        # Price distance to pivot (as % of ATR)
        if not np.isnan(s3) and atr_14[i] > 0:
            dist_to_s3 = (close[i] - s3) / atr_14[i]
            dist_to_r3 = (r3 - close[i]) / atr_14[i]
        else:
            dist_to_s3 = 999
            dist_to_r3 = 999
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # LONG: Price at S3 support + 1w uptrend + volume spike
        # S3 touch = dist_to_s3 between -0.3 and +1.0 ATR (price just touched or slightly above S3)
        if is_bullish_1w and dist_to_s3 >= -0.3 and dist_to_s3 <= 1.0:
            if vol_spike:
                desired_signal = SIZE
        
        # SHORT: Price at R3 resistance + 1w downtrend + volume spike
        if not is_bullish_1w and dist_to_r3 >= -0.3 and dist_to_r3 <= 1.0:
            if vol_spike:
                desired_signal = -SIZE
        
        # === STOPLOSS CHECK ===
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.0 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.0 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                desired_signal = 0.0
        
        # === TAKE PROFIT at opposite pivot ===
        if in_position and position_side > 0:
            # TP when price reaches R3
            if not np.isnan(r3) and high[i] >= r3:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # TP when price reaches S3
            if not np.isnan(s3) and low[i] <= s3:
                desired_signal = 0.0
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                if position_side > 0:
                    stop_price = close[i] - 2.0 * entry_atr
                else:
                    stop_price = close[i] + 2.0 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_atr = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = desired_signal
    
    return signals