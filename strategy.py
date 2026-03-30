#!/usr/bin/env python3
"""
Experiment #021: 4h Camarilla S4/R4 + Volume Spike 2.0x + 1d EMA200 Trend

HYPOTHESIS: camarilla S4/R4 are institutional support/resistance levels.
By using stricter volume spike (2.0x vs 1.5x) + 1d EMA200 trend filter,
this reduces #009's 275 trades to target range 100-150.

WHY IT WORKS IN BULL AND BEAR: Symmetrical pivot levels — buy S3/S4 in uptrends,
short R3/R4 in downtrends. Works in both bull rallies and bear bounces.

TARGET: 100-150 total trades over 4 years. Signal size: 0.30.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_camarilla_vol2_ema200_1d_v1"
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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA200 for trend direction
    ema_200_1d = pd.Series(df_1d['close'].values).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema200_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # === Local 4h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Volume ratio (strict 2.0x threshold)
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_sma > 0, vol_sma, 1)
    
    # Signals
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_bar = 0
    
    # Warmup: ATR(14) + EMA200(1d) alignment
    warmup = 1500
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(ema200_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === TREND DIRECTION (1d EMA200) ===
        price_above_ema = close[i] > ema200_aligned[i]
        
        # === CAMARILLA LEVELS from previous CLOSED bar (no look-ahead) ===
        prev_high = high[i - 1]
        prev_low = low[i - 1]
        prev_close = close[i - 1]
        prev_range = prev_high - prev_low
        
        # Classic Camarilla levels
        r3 = prev_close + prev_range * 0.09167
        r4 = prev_close + prev_range * 0.18333
        s3 = prev_close - prev_range * 0.09167
        s4 = prev_close - prev_range * 0.18333
        
        desired_signal = 0.0
        
        if not in_position:
            # === LONG: Price at S3/S4 with volume spike + uptrend ===
            if price_above_ema and vol_ratio[i] > 2.0:
                if low[i] <= s4:
                    desired_signal = SIZE
                elif low[i] <= s3:
                    desired_signal = SIZE
            
            # === SHORT: Price at R3/R4 with volume spike + downtrend ===
            elif not price_above_ema and vol_ratio[i] > 2.0:
                if high[i] >= r4:
                    desired_signal = -SIZE
                elif high[i] >= r3:
                    desired_signal = -SIZE
        
        else:
            # === TRAILING STOP (2.5 ATR) ===
            if position_side > 0:
                highest_since_entry = max(highest_since_entry, high[i])
                trailing_stop = highest_since_entry - 2.5 * atr_14[i]
                if low[i] < trailing_stop:
                    desired_signal = 0.0
            
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, low[i])
                trailing_stop = lowest_since_entry + 2.5 * atr_14[i]
                if high[i] > trailing_stop:
                    desired_signal = 0.0
            
            # === MINIMUM HOLD (2 bars) + take profit at prev close ===
            bars_held = i - entry_bar
            if bars_held >= 2:
                if position_side > 0 and close[i] >= prev_close:
                    desired_signal = 0.0
                elif position_side < 0 and close[i] <= prev_close:
                    desired_signal = 0.0
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                entry_bar = i
        else:
            if in_position:
                in_position = False
                position_side = 0
        
        signals[i] = desired_signal
    
    return signals