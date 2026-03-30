#!/usr/bin/env python3
"""
Experiment #024: 6h Bollinger-Keltner Squeeze Breakout + 1d SMA + Volume

HYPOTHESIS: When Bollinger Bands contract INSIDE Keltner Channels (squeeze),
volatility compresses. A subsequent expansion with volume spike indicates
institutional accumulation/distribution. This works in both directions:
- Bull: squeeze in uptrend + bullish volume expansion = strong signal
- Bear: squeeze in downtrend + bearish volume expansion = short signal
- Range: squeeze breakouts often false, filtered by 1d SMA

Why 6h: captures institutional activity patterns without overtrading.
BB(20,2) + KC(20,1.5) is the classic TTM squeeze indicator.

TARGET: 75-150 total over 4 years (18-37/year). Size: 0.30.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_bb_kc_squeeze_1d_vol_v1"
timeframe = "6h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < 2:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_bollinger_bands(close, period=20, num_std=2):
    """Bollinger Bands"""
    mid = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    upper = mid + num_std * std
    lower = mid - num_std * std
    return upper, mid, lower

def calculate_keltner(high, low, close, period=20, multiplier=1.5):
    """Keltner Channels using ATR"""
    n = len(close)
    atr = calculate_atr(high, low, close, period)
    mid = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    upper = mid + multiplier * atr
    lower = mid - multiplier * atr
    return upper, mid, lower

def calculate_donchian(high, low, period=20):
    """Donchian channel for momentum"""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d SMA200 for macro trend (call ONCE) ===
    df_1d = get_htf_data(prices, '1d')
    sma_1d_200 = pd.Series(df_1d['close'].values).rolling(window=200, min_periods=200).mean().values
    sma_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_1d_200)
    
    # === 6h Indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Bollinger Bands (20, 2)
    bb_upper, bb_mid, bb_lower = calculate_bollinger_bands(close, period=20, num_std=2)
    
    # Keltner Channels (20, 1.5)
    kc_upper, kc_mid, kc_lower = calculate_keltner(high, low, close, period=20, multiplier=1.5)
    
    # Donchian for momentum confirmation
    dc_upper_20, dc_lower_20 = calculate_donchian(high, low, period=20)
    
    # Volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 1e-10, vol_ma, 1.0)
    
    # === SQEEZE DETECTION ===
    # Squeeze ON: BB inside KC (BB upper < KC upper AND BB lower > KC lower)
    # This means low volatility - potential breakout coming
    squeeze_on = (bb_upper < kc_upper) & (bb_lower > kc_lower)
    
    # Squeeze OFF: BB breaks outside KC = volatility expansion
    squeeze_off_up = (bb_upper > kc_upper) & (bb_mid > bb_mid[i-1] if i > 0 else False)
    
    # === Signals ===
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    squeeze_was_on = False
    
    warmup = 100  # Need enough for BB(20) + SMA200(1d)
    
    for i in range(warmup, n):
        # NaN check
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(sma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Track squeeze state
        is_squeeze_now = squeeze_on[i]
        
        # === TREND DETECTION (1d SMA200) ===
        htf_bullish = close[i] > sma_1d_aligned[i]
        htf_bearish = close[i] < sma_1d_aligned[i]
        
        # Local momentum (Donchian position)
        dc_mid = (dc_upper_20[i] + dc_lower_20[i]) / 2 if not np.isnan(dc_upper_20[i]) else close[i]
        local_bullish = close[i] > dc_mid
        local_bearish = close[i] < dc_mid
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.8
        
        # === SQUEEZE BREAKOUT LOGIC ===
        # Long: squeeze was ON, now expanding UP with volume
        squeeze_expansion_up = is_squeeze_now == False and squeeze_was_on and close[i] > kc_upper[i]
        # Short: squeeze was ON, now expanding DOWN with volume
        squeeze_expansion_down = is_squeeze_now == False and squeeze_was_on and close[i] < kc_lower[i]
        
        # === Update position tracking ===
        if in_position:
            if position_side > 0:
                highest_since_entry = max(highest_since_entry, high[i])
            else:
                lowest_since_entry = min(lowest_since_entry, low[i])
        
        # === ATR TRAILING STOP (2.5x ATR) ===
        def check_atr_stop():
            if not in_position:
                return False
            if position_side > 0:
                return low[i] < (highest_since_entry - 2.5 * entry_atr)
            else:
                return high[i] > (lowest_since_entry + 2.5 * entry_atr)
        
        # === MIN HOLD: 2 bars (12h) ===
        min_hold = (i - entry_bar) >= 2
        
        # === EXITS ===
        if in_position:
            stop_hit = check_atr_stop()
            
            # Exit on opposite trend + min hold
            if position_side > 0 and htf_bearish and min_hold:
                stop_hit = True
            if position_side < 0 and htf_bullish and min_hold:
                stop_hit = True
            
            # Exit on Donchian reversal (price crosses midpoint)
            if position_side > 0 and close[i] < dc_mid and min_hold:
                stop_hit = True
            if position_side < 0 and close[i] > dc_mid and min_hold:
                stop_hit = True
            
            if stop_hit:
                signals[i] = 0.0
                in_position = False
                position_side = 0
            else:
                signals[i] = position_side * SIZE
        
        # === NEW POSITIONS ===
        if not in_position:
            # LONG: Squeeze expansion UP + volume + 1d uptrend + local bullish
            if squeeze_expansion_up and vol_spike and htf_bullish and local_bullish:
                in_position = True
                position_side = 1
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                highest_since_entry = high[i]
                signals[i] = SIZE
            
            # LONG (no squeeze): Price above KC upper + strong volume + 1d uptrend
            elif close[i] > kc_upper[i] and vol_spike and htf_bullish:
                in_position = True
                position_side = 1
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                highest_since_entry = high[i]
                signals[i] = SIZE * 0.7  # Smaller size - no squeeze confirmation
            
            # SHORT: Squeeze expansion DOWN + volume + 1d downtrend + local bearish
            elif squeeze_expansion_down and vol_spike and htf_bearish and local_bearish:
                in_position = True
                position_side = -1
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                lowest_since_entry = low[i]
                signals[i] = -SIZE
            
            # SHORT (no squeeze): Price below KC lower + strong volume + 1d downtrend
            elif close[i] < kc_lower[i] and vol_spike and htf_bearish:
                in_position = True
                position_side = -1
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                lowest_since_entry = low[i]
                signals[i] = -SIZE * 0.7  # Smaller size - no squeeze confirmation
            
            else:
                signals[i] = 0.0
        
        # Store squeeze state for next iteration
        squeeze_was_on = is_squeeze_now
    
    return signals