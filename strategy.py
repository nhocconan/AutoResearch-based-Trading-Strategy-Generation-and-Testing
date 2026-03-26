#!/usr/bin/env python3
"""
Experiment #021: 4h Camarilla Bounce + Choppiness Regime + Volume

HYPOTHESIS: Top DB performer (ETH test Sharpe 1.47, 95 trades) uses Camarilla
S3/R3 bounce logic in choppy markets. Price bounces off extreme Camarilla levels
in range-bound conditions, fading the move. This is fundamentally DIFFERENT from
breakout trading - it profits from mean-reversion, not trend continuation.

KEY INSIGHT: Donchian breakout strategies all failed. Camarilla bounce strategy
succeeded because:
1. It trades extremes, not breakouts (higher probability)
2. Choppiness filter prevents trading in trending markets
3. Volume confirmation ensures institutional interest

TIMEFRAME: 4h primary
HTF: 12h for regime confirmation
TARGET: 75-150 total trades over 4 years (19-37/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_camarilla_bounce_chop_vol_v1"
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

def calculate_choppiness_index(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    CHOP > 61.8 = choppy/range-bound market (mean reversion favored)
    CHOP < 38.2 = trending market (trend following favored)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    chop = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        # Sum of ATR over period
        atr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            tr = max(high[j] - low[j], abs(high[j] - close[j-1]) if j > 0 else high[j] - low[j])
            atr_sum += tr
        
        # Highest high - lowest low over period
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        range_hl = highest_high - lowest_low
        
        if range_hl > 0 and atr_sum > 0:
            # CHOP = 100 * log10(atr_sum / range_hl) / log10(period)
            chop[i] = 100 * np.log10(atr_sum / range_hl) / np.log10(period)
    
    return chop

def calculate_camarilla_levels(high, low, close):
    """
    Camarilla Pivot Levels
    R3 = close + (high - low) * 1.1 / 2
    R4 = close + (high - low) * 1.1
    S3 = close - (high - low) * 1.1 / 2
    S4 = close - (high - low) * 1.1
    """
    n = len(close)
    r3 = np.full(n, np.nan, dtype=np.float64)
    r4 = np.full(n, np.nan, dtype=np.float64)
    s3 = np.full(n, np.nan, dtype=np.float64)
    s4 = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(1, n):
        hl_range = high[i] - low[i]
        r3[i] = close[i] + hl_range * 1.1 / 2
        r4[i] = close[i] + hl_range * 1.1
        s3[i] = close[i] - hl_range * 1.1 / 2
        s4[i] = close[i] - hl_range * 1.1
    
    return r3, r4, s3, s4

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE ===
    df_12h = get_htf_data(prices, '12h')
    
    # 12h Choppiness for regime (smoother, less noise)
    chop_12h_raw = calculate_choppiness_index(df_12h['high'].values, df_12h['low'].values, 
                                               df_12h['close'].values, period=14)
    chop_12h_aligned = align_htf_to_ltf(prices, df_12h, chop_12h_raw)
    
    # Calculate local 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # 4h Choppiness
    chop_4h = calculate_choppiness_index(high, low, close, period=14)
    
    # Camarilla levels
    r3, r4, s3, s4 = calculate_camarilla_levels(high, low, close)
    
    # Volume MA for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # RSI for additional confirmation
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(span=14, min_periods=14, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = (100 - (100 / (1 + rs))).values
    
    # ATR-based stops
    atr_ma = pd.Series(atr_14).rolling(window=20, min_periods=20).mean().values
    atr_percent = atr_14 / close * 100
    
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
    
    warmup = 50
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(chop_12h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME CHECK ===
        # CHOP > 50 means choppy (no clear trend) - perfect for Camarilla bounces
        # Use average of 4h and 12h CHOP for confirmation
        chop_val = np.nanmean([chop_4h[i] if not np.isnan(chop_4h[i]) else 50, 
                               chop_12h_aligned[i]])
        is_choppy = chop_val > 50.0  # Range-bound market
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.2
        
        # === RSI FOR MOMENTUM ===
        rsi_val = rsi[i]
        
        # === Camarilla bounce conditions ===
        # Price touched or approached R3/R4 (overbought) = potential short
        near_r3 = abs(close[i] - r3[i]) < 0.5 * atr_14[i] if not np.isnan(r3[i]) else False
        near_r4 = abs(close[i] - r4[i]) < 0.5 * atr_14[i] if not np.isnan(r4[i]) else False
        
        # Price touched or approached S3/S4 (oversold) = potential long
        near_s3 = abs(close[i] - s3[i]) < 0.5 * atr_14[i] if not np.isnan(s3[i]) else False
        near_s4 = abs(close[i] - s4[i]) < 0.5 * atr_14[i] if not np.isnan(s4[i]) else False
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === NEW LONG ENTRY (bounce off S3/S4) ===
            # Conditions: near support level + RSI oversold + volume + choppy
            if (near_s3 or near_s4) and rsi_val < 40 and vol_spike and is_choppy:
                desired_signal = SIZE
            
            # === NEW SHORT ENTRY (bounce off R3/R4) ===
            # Conditions: near resistance level + RSI overbought + volume + choppy
            if (near_r3 or near_r4) and rsi_val > 60 and vol_spike and is_choppy:
                desired_signal = -SIZE
        
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
        
        # === EXIT: Mean reversion target or ATR profit ===
        exit_triggered = False
        
        if in_position and position_side > 0:
            # Long exit: RSI normalized or touched middle (R3)
            if rsi_val > 55:
                exit_triggered = True
            # Or if RSI reached neutral zone (40-60)
            if 40 <= rsi_val <= 60:
                exit_triggered = True
            # Or price reached R3 (mean reversion complete)
            if not np.isnan(r3[i]) and close[i] >= r3[i]:
                exit_triggered = True
        
        if in_position and position_side < 0:
            # Short exit: RSI normalized
            if rsi_val < 45:
                exit_triggered = True
            # Or if RSI reached neutral zone
            if 40 <= rsi_val <= 60:
                exit_triggered = True
            # Or price reached S3
            if not np.isnan(s3[i]) and close[i] <= s3[i]:
                exit_triggered = True
        
        if exit_triggered:
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
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
            else:
                # Same direction - maintain position
                pass
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