#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h 123 Reversal Pattern + Volume Confirmation
# Uses the 1-2-3 reversal pattern (Ross Hook) to identify trend exhaustion and reversals:
# 1. Price makes a new high/low (point 1)
# 2. Pullback to form swing point (point 2)  
# 3. Failed attempt to exceed point 1 (point 3) triggers reversal
# Works in both bull and bear markets by capturing reversals at trend extremes.
# Target: 60-120 total trades over 4 years (15-30/year) with high-probability reversals.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 6h data (primary timeframe) ===
    df_6h = get_htf_data(prices, '6h')
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    volume_6h = df_6h['volume'].values
    
    # === 1d data (higher timeframe for trend filter) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === Calculate swing points for 1-2-3 pattern ===
    # Find swing highs and lows using 5-bar window
    def find_swing_points(high_arr, low_arr, window=5):
        n = len(high_arr)
        swing_high = np.full(n, np.nan)
        swing_low = np.full(n, np.nan)
        
        for i in range(window, n - window):
            # Swing high: highest high in window
            if high_arr[i] == np.max(high_arr[i-window:i+window+1]):
                swing_high[i] = high_arr[i]
            # Swing low: lowest low in window
            if low_arr[i] == np.min(low_arr[i-window:i+window+1]):
                swing_low[i] = low_arr[i]
        return swing_high, swing_low
    
    swing_high_6h, swing_low_6h = find_swing_points(high_6h, low_6h, 5)
    
    # === 123 pattern detection ===
    # Bullish 123: 
    # 1. Point 1: swing low
    # 2. Point 2: pullback high (higher than point 1)
    # 3. Point 3: failed attempt to make new low (higher than point 2)
    # Bearish 123:
    # 1. Point 1: swing high
    # 2. Point 2: pullback low (lower than point 1)
    # 3. Point 3: failed attempt to make new high (lower than point 2)
    
    bullish_setup = np.zeros(n, dtype=bool)
    bearish_setup = np.zeros(n, dtype=bool)
    
    for i in range(10, n):  # Need sufficient lookback
        # Look for bullish 123 pattern
        # Find most recent swing low (point 1)
        point1_idx = None
        for j in range(i-20, i):
            if j >= 0 and not np.isnan(swing_low_6h[j]):
                point1_idx = j
                break
        
        if point1_idx is not None:
            point1_val = swing_low_6h[point1_idx]
            # Find pullback high after point 1 (point 2)
            point2_idx = None
            for j in range(point1_idx+1, i):
                if high_6h[j] > point1_val * 1.005:  # At least 0.5% above point 1
                    point2_idx = j
                    break
            
            if point2_idx is not None:
                point2_val = high_6h[point2_idx]
                # Check for failed attempt to make new low (point 3)
                # Price should not go below point 2 in recent bars
                recent_low = np.min(low_6h[point2_idx:i+1])
                if recent_low > point2_val * 0.995:  # Held above point 2
                    bullish_setup[i] = True
        
        # Look for bearish 123 pattern
        # Find most recent swing high (point 1)
        point1_idx = None
        for j in range(i-20, i):
            if j >= 0 and not np.isnan(swing_high_6h[j]):
                point1_idx = j
                break
        
        if point1_idx is not None:
            point1_val = swing_high_6h[point1_idx]
            # Find pullback low after point 1 (point 2)
            point2_idx = None
            for j in range(point1_idx+1, i):
                if low_6h[j] < point1_val * 0.995:  # At least 0.5% below point 1
                    point2_idx = j
                    break
            
            if point2_idx is not None:
                point2_val = low_6h[point2_idx]
                # Check for failed attempt to make new high (point 3)
                # Price should not go above point 2 in recent bars
                recent_high = np.max(high_6h[point2_idx:i+1])
                if recent_high < point2_val * 1.005:  # Held below point 2
                    bearish_setup[i] = True
    
    # === Volume confirmation ===
    vol_ma_20_6h = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_6h > (1.5 * vol_ma_20_6h)
    
    # === 1d trend filter (avoid trading against strong trend) ===
    # Use 20-period EMA on 1d to determine trend direction
    ema_20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    # Uptrend when close > EMA20, downtrend when close < EMA20
    uptrend_filter = close_6h > ema_20_aligned
    downtrend_filter = close_6h < ema_20_aligned
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 30
    
    # Track position and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(vol_ma_20_6h[i]) or 
            np.isnan(ema_20_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        bullish_setup_val = bullish_setup[i]
        bearish_setup_val = bearish_setup[i]
        vol_spike_val = vol_spike[i]
        is_uptrend = uptrend_filter[i]
        is_downtrend = downtrend_filter[i]
        
        # === STOPLOSS LOGIC ===
        if position == 1:  # Long position
            atr_6h = np.abs(high_6h - low_6h)
            atr_ma = pd.Series(atr_6h).rolling(window=10, min_periods=10).mean().values
            atr_aligned = align_htf_to_ltf(prices, df_6h, atr_ma)
            atr_val = atr_aligned[i]
            if price < entry_price - 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        elif position == -1:  # Short position
            atr_6h = np.abs(high_6h - low_6h)
            atr_ma = pd.Series(atr_6h).rolling(window=10, min_periods=10).mean().values
            atr_aligned = align_htf_to_ltf(prices, df_6h, atr_ma)
            atr_val = atr_aligned[i]
            if price > entry_price + 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when bearish setup appears or price drops below point 2 of pattern
            if bearish_setup_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        elif position == -1:  # Short position
            # Exit when bullish setup appears or price rises above point 2 of pattern
            if bullish_setup_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Bullish 123 with volume confirmation in uptrend or ranging market
            if bullish_setup_val and vol_spike_val and (is_uptrend or not is_downtrend):
                signals[i] = 0.25
                position = 1
                entry_price = price
                continue
            # Bearish 123 with volume confirmation in downtrend or ranging market
            elif bearish_setup_val and vol_spike_val and (is_downtrend or not is_uptrend):
                signals[i] = -0.25
                position = -1
                entry_price = price
                continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_123Reversal_Pattern_VolumeFilter_v1"
timeframe = "6h"
leverage = 1.0