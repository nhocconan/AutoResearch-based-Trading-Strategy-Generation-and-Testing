#!/usr/bin/env python3
"""
Experiment #012: 12h Bollinger Squeeze Breakout + Volume + ATR

HYPOTHESIS: Bollinger Band squeeze (width at 30d low) marks volatility compression
before explosive moves. Price breaking outside BB after squeeze identifies the
"pop" phase. Volume confirms institutional participation. Works in both bull
(breakout up) and bear (breakout down) markets.

TIMEFRAME: 12h primary
HTF: 1d SMA for trend bias
TARGET: 75-150 total trades over 4 years (19-37/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_bb_squeeze_vol_atr_v1"
timeframe = "12h"
leverage = 1.0

def calculate_bb_width_pct(high, low, close, period=20, std_mult=2.0):
    """Bollinger Band width as percentage of middle band"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    # BB middle (SMA)
    middle = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    
    # BB standard deviation
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = middle + std_mult * std
    lower = middle - std_mult * std
    
    # Width as percentage of middle band
    width = (upper - lower) / (middle + 1e-10)
    
    return width

def calculate_bb_width_percentile(width, period=30):
    """Percentile rank of BB width over recent period"""
    n = len(width)
    if n < period:
        return np.full(n, np.nan)
    
    result = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        if not np.isnan(width[i]):
            window = width[max(0, i - period + 1):i + 1]
            valid = window[~np.isnan(window)]
            if len(valid) > 0:
                result[i] = (np.sum(valid < width[i]) / len(valid)) * 100
    
    return result

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
    
    # === Load HTF data ONCE ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d SMA for trend bias
    sma_1d = pd.Series(df_1d['close'].values).rolling(window=21, min_periods=21).mean().values
    sma_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_1d)
    
    # Calculate local 12h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Bollinger Bands
    period = 20
    middle = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    bb_upper = middle + 2.0 * std
    bb_lower = middle - 2.0 * std
    
    # BB Width percentile (squeeze detection)
    bb_width = calculate_bb_width_pct(high, low, close, period=20)
    bb_width_pct = calculate_bb_width_percentile(bb_width, period=30)
    
    # Volume MA
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
    entry_bar = 0
    target_price = 0.0
    
    warmup = 50
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(sma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === TREND BIAS (1d SMA) ===
        htf_bullish = close[i] > sma_1d_aligned[i]
        
        # === SQEEZE DETECTION ===
        # Squeeze = BB width at low percentile (volatility compressed)
        squeeze = bb_width_pct[i] < 25  # width at bottom 25% of range
        
        # === BREAKOUT DETECTION ===
        # Price breaks above upper BB
        breakout_up = close[i] > bb_upper[i]
        # Price breaks below lower BB
        breakout_down = close[i] < bb_lower[i]
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.4
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG ENTRY ===
            # Breakout above upper BB after squeeze
            if breakout_up:
                # Volume confirmation OR trend aligned (looser)
                if vol_spike or htf_bullish:
                    desired_signal = SIZE
            
            # === SHORT ENTRY ===
            # Breakout below lower BB after squeeze
            if breakout_down:
                if vol_spike or not htf_bullish:
                    desired_signal = -SIZE
        
        # === STOPLOSS CHECK (3 ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 3.0 * atr_14[i]
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 3.0 * atr_14[i]
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === TAKE PROFIT (2:1 R:R) ===
        tp_triggered = False
        
        if in_position and position_side > 0:
            if close[i] >= target_price and target_price > 0:
                tp_triggered = True
        
        if in_position and position_side < 0:
            if close[i] <= target_price and target_price > 0:
                tp_triggered = True
        
        if tp_triggered:
            # Exit with profit
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
                
                # Set stoploss and target
                if position_side > 0:
                    stop_price = entry_price - 3.0 * entry_atr
                    target_price = entry_price + 4.5 * entry_atr  # 1.5R
                else:
                    stop_price = entry_price + 3.0 * entry_atr
                    target_price = entry_price - 4.5 * entry_atr  # 1.5R
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
                entry_bar = 0
                target_price = 0.0
        
        signals[i] = desired_signal
    
    return signals