#!/usr/bin/env python3
"""
Experiment #021: 4h Camarilla S3/S4 + Volume Spike + Choppiness Regime

HYPOTHESIS: Camarilla S3/S4 levels are institutional support/resistance where
smart money accumulates/distributes. By entering on touches of these levels WITH
volume spike AND when market is in trending regime (CHOP < 38.2), we catch
high-probability reversals.

WHY IT WORKS IN BULL AND BEAR: Symmetrical pivot formula — buy S3/S4 dips
in uptrends, short R3/R4 rallies in downtrends. CHOP filter avoids ranging
markets where Camarilla levels fail.

FROM DB: gen_camarilla_pivot_volume_spike_choppiness_4h_v1 achieved
test_sharpe=1.471 with 95 trades on ETHUSDT. This strategy adapts that pattern.

TARGET: 75-150 total trades over 4 years (19-37/year).
Signal size: 0.30.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_camarilla_vol_chop_regime_1d_v1"
timeframe = "4h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Average True Range using EWM"""
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
    """Choppiness Index - values < 38.2 indicate trending, > 61.8 indicate choppy"""
    n = len(close)
    chop = np.full(n, np.nan)
    
    for i in range(period, n):
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        
        if highest_high == lowest_low:
            chop[i] = 50.0
            continue
        
        period_sum = 0.0
        for j in range(i - period + 1, i + 1):
            period_sum += high[j] - low[j]
        
        chop[i] = 100.0 * (np.log(period_sum) / np.log(highest_high - lowest_low)) if (highest_high - lowest_low) > 0 else 50.0
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA50 for trend direction (filter out countertrend entries)
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # === Local 4h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    
    # Volume ratio (20-bar SMA)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # === Get previous day's OHLC (for Camarilla levels) ===
    # Each 4h bar maps to a 1d bar - use the 1d dataframe
    prev_day_high = df_1d['high'].values
    prev_day_low = df_1d['low'].values
    prev_day_close = df_1d['close'].values
    
    # Align 1d data to 4h bars
    prev_high_aligned = align_htf_to_ltf(prices, df_1d, prev_day_high)
    prev_low_aligned = align_htf_to_ltf(prices, df_1d, prev_day_low)
    prev_close_aligned = align_htf_to_ltf(prices, df_1d, prev_day_close)
    
    # === Signals ===
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
    entry_bar = 0
    
    warmup = 150  # Need enough for EMA50 + CHOP alignment buffer
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(chop[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(ema_1d_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === REGIME FILTER: Only trade when CHOP < 38.2 (trending) ===
        is_trending = chop[i] < 38.2
        
        # === TREND DIRECTION (1d EMA50) ===
        price_above_1d_ema = close[i] > ema_1d_aligned[i]
        
        # === CAMARILLA LEVELS from previous day's range ===
        day_high = prev_high_aligned[i]
        day_low = prev_low_aligned[i]
        day_close = prev_close_aligned[i]
        day_range = day_high - day_low
        
        # Skip if invalid day data
        if np.isnan(day_high) or np.isnan(day_low) or day_range <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # Classic Camarilla factors
        r3 = day_close + day_range * 0.09167
        r4 = day_close + day_range * 0.18333
        s3 = day_close - day_range * 0.09167
        s4 = day_close - day_range * 0.18333
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position and is_trending:
            # === LONG: Price touches S3 or S4 with volume confirmation ===
            if price_above_1d_ema and vol_spike:
                # S4 touch (deeper level = better R/R)
                if low[i] <= s4:
                    desired_signal = SIZE
                # S3 touch
                elif low[i] <= s3:
                    desired_signal = SIZE
            
            # === SHORT: Price touches R3 or R4 with volume confirmation ===
            if not price_above_1d_ema and vol_spike:
                # R4 touch
                if high[i] >= r4:
                    desired_signal = -SIZE
                # R3 touch
                elif high[i] >= r3:
                    desired_signal = -SIZE
        
        # === MINIMUM HOLD: 2 bars to avoid fee churn ===
        bars_held = i - entry_bar
        
        # === TRAILING STOP (2.0 ATR) ===
        if in_position:
            if position_side > 0:
                highest_since_entry = max(highest_since_entry, high[i])
                trailing_stop = highest_since_entry - 2.0 * entry_atr
                stop_price = max(stop_price, trailing_stop)
                # Exit on stop hit
                if low[i] < stop_price:
                    desired_signal = 0.0
            
            if position_side < 0:
                lowest_since_entry = min(lowest_since_entry, low[i])
                trailing_stop = lowest_since_entry + 2.0 * entry_atr
                stop_price = min(stop_price, trailing_stop)
                # Exit on stop hit
                if high[i] > stop_price:
                    desired_signal = 0.0
        
        # === TAKE PROFIT at day midpoint (after min hold) ===
        if in_position and bars_held >= 2:
            midpoint = (day_high + day_low) / 2.0
            if position_side > 0 and close[i] >= midpoint:
                desired_signal = 0.0
            if position_side < 0 and close[i] <= midpoint:
                desired_signal = 0.0
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New entry or direction flip
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                entry_bar = i
                if position_side > 0:
                    stop_price = entry_price - 2.0 * entry_atr
                else:
                    stop_price = entry_price + 2.0 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
        
        signals[i] = desired_signal
    
    return signals