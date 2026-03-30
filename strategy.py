#!/usr/bin/env python3
"""
Experiment #028: 4h Camarilla S3/R3 Mean Reversion + Volume + Choppiness

HYPOTHESIS: Camarilla S3/R3 are statistical extreme levels derived from decades of 
floor trading observation. Price reaching S3 (oversold) or R3 (overbought) with 
volume confirmation and non-choppy regime creates high-probability mean-reversion 
setups that work in both bull and bear markets.

WHY IT WORKS: Camarilla formula: R3 = Close + (H-L)*1.1/2 + (H-L)*1.1/4.
S3/R3 are the extreme bands. When price reaches these levels with volume spike,
institutions are often reversing positions. Choppiness <61.8 ensures we don't
fade moves in ranging markets. 1d SMA200 confirms trend direction alignment.

TARGET: 50-150 total trades over 4 years (12-37/year). HARD MAX: 300.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_camarilla_vol_chop_regime_v1"
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

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index - lower = trending, higher = choppy"""
    n = len(high)
    chop = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        tr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            if j > 0:
                tr = max(high[j] - low[j], abs(high[j] - close[j-1]))
            else:
                tr = high[j] - low[j]
            tr_sum += tr
        
        if tr_sum > 0:
            hh = np.max(high[i - period + 1:i + 1])
            ll = np.min(low[i - period + 1:i + 1])
            range_hl = hh - ll
            
            if range_hl > 0:
                chop[i] = 100 * np.log10(tr_sum / range_hl) / np.log10(period)
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d SMA200 for trend direction
    sma_200_1d = pd.Series(df_1d['close'].values).rolling(window=200, min_periods=200).mean().values
    sma_200_aligned = align_htf_to_ltf(prices, df_1d, sma_200_1d)
    
    # Local 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    
    # Volume SMA for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # === Pre-compute Camarilla levels from aligned 1d data ===
    # Use previous 1d bar's OHLC for pivot calculation
    prev_close_1d = np.roll(df_1d['close'].values, 1)  # Previous day close
    prev_high_1d = np.roll(df_1d['high'].values, 1)     # Previous day high
    prev_low_1d = np.roll(df_1d['low'].values, 1)      # Previous day low
    
    # Set first element to current (no previous data available)
    prev_close_1d[0] = df_1d['close'].values[0]
    prev_high_1d[0] = df_1d['high'].values[0]
    prev_low_1d[0] = df_1d['low'].values[0]
    
    # Camarilla S3/R3 calculation
    hl_range = prev_high_1d - prev_low_1d
    
    # R3 = Close + H/2 + H/4 (classic formula)
    r3_1d = prev_close_1d + hl_range * 1.5 / 2 + hl_range * 1.5 / 4
    # S3 = Close - H/2 - H/4
    s3_1d = prev_close_1d - hl_range * 1.5 / 2 - hl_range * 1.5 / 4
    
    # R4 = Close + H/2 + H/4 + H/6
    r4_1d = prev_close_1d + hl_range * 1.5 / 2 + hl_range * 1.5 / 4 + hl_range * 1.5 / 6
    # S4 = Close - H/2 - H/4 - H/6
    s4_1d = prev_close_1d - hl_range * 1.5 / 2 - hl_range * 1.5 / 4 - hl_range * 1.5 / 6
    
    # Pivot point (PP) = (H + L + Close) / 3
    pivot_1d = (prev_high_1d + prev_low_1d + prev_close_1d) / 3
    
    # Align to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    
    signals = np.zeros(n)
    SIZE = 0.25  # Position size
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_bar = 0
    stop_price = 0.0
    
    warmup = 250  # Need 200 for SMA200 + buffer
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(sma_200_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(chop[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # Get Camarilla levels for this bar
        curr_r3 = r3_aligned[i] if not np.isnan(r3_aligned[i]) else 0
        curr_s3 = s3_aligned[i] if not np.isnan(s3_aligned[i]) else 0
        curr_r4 = r4_aligned[i] if not np.isnan(r4_aligned[i]) else 0
        curr_s4 = s4_aligned[i] if not np.isnan(s4_aligned[i]) else 0
        curr_pivot = pivot_aligned[i] if not np.isnan(pivot_aligned[i]) else close[i]
        
        # === TREND DIRECTION (1d SMA200) ===
        price_above_200 = close[i] > sma_200_aligned[i]
        price_below_200 = close[i] < sma_200_aligned[i]
        
        # === REGIME (Choppiness Index) ===
        # Only trade when not too choppy (CHOP < 61.8)
        is_choppy = chop[i] > 61.8
        
        # Skip if too choppy
        if is_choppy and not in_position:
            signals[i] = 0.0
            continue
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # Previous bar's close for breakout detection
        prev_close = close[i - 1]
        
        desired_signal = 0.0
        
        # === ENTRY LOGIC ===
        if not in_position:
            # === LONG: Price reaches S3 level (oversold extreme) ===
            # S3 is statistical oversold - expect bounce
            # Requirements: price below S3, volume spike, not choppy, trend alignment
            if curr_s3 > 0:
                # Price touches or crosses below S3
                price_at_s3_zone = low[i] <= curr_s3 + 0.5 * atr_14[i]  # Within ATR of S3
                
                if price_at_s3_zone and vol_spike and price_above_200:
                    # Bull trend + oversold + volume = strong bounce setup
                    desired_signal = SIZE
            
            # === LONG: Price breaks below S4 (extreme oversold) ===
            if curr_s4 > 0:
                price_below_s4 = low[i] < curr_s4
                if price_below_s4 and vol_spike:
                    # Extreme oversold - mean revert
                    desired_signal = SIZE
            
            # === SHORT: Price reaches R3 level (overbought extreme) ===
            # R3 is statistical overbought - expect pullback
            if curr_r3 > 0:
                price_at_r3_zone = high[i] >= curr_r3 - 0.5 * atr_14[i]  # Within ATR of R3
                
                if price_at_r3_zone and vol_spike and price_below_200:
                    # Bear trend + overbought + volume = strong reversal setup
                    desired_signal = -SIZE
            
            # === SHORT: Price breaks above R4 (extreme overbought) ===
            if curr_r4 > 0:
                price_above_r4 = high[i] > curr_r4
                if price_above_r4 and vol_spike:
                    # Extreme overbought - mean revert
                    desired_signal = -SIZE
        
        # === STOPLOSS CHECK (2.0 ATR) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.0 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.0 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === MEAN REVERSION EXIT (price returns to pivot) ===
        if in_position:
            bars_held = i - entry_bar
            min_hold = 4  # Hold at least 4 bars (16h)
            
            if bars_held >= min_hold:
                # Exit long when price reaches pivot
                if position_side > 0 and close[i] >= curr_pivot:
                    desired_signal = 0.0
                
                # Exit short when price reaches pivot
                if position_side < 0 and close[i] <= curr_pivot:
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
                entry_bar = i
                if position_side > 0:
                    stop_price = entry_price - 2.0 * entry_atr
                else:
                    stop_price = entry_price + 2.0 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = desired_signal
    
    return signals