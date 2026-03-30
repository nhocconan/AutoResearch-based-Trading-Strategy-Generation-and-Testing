#!/usr/bin/env python3
"""
Experiment #007: 6h Ichimoku TK/Kijun Cross + 1d Cloud Filter + Choppiness

HYPOTHESIS: 
- Tenkan/Kijun cross (fast/slow EMA equivalent in Ichimoku) for entries
- 1d Cloud (Senkou A/B) as structural filter - price must be on correct side
- Choppiness < 55 to avoid ranging markets
- Volume confirmation to avoid false breakouts

WHY NOVEL:
- Different from all tested strategies (HMA cross, DEMA cross, Donchian)
- Ichimoku is specifically designed to work in both trending and ranging markets
- 1d cloud is SLOWER structure than 6h - gives reliable trend direction
- TK/Kijun cross is faster than price but slower than pure EMA cross

EXPECTED TRADE COUNT:
- 6h ~35000 bars over 4 years
- TK cross triggers ~30-50 times/year without filters
- With 1d cloud filter + chop: ~15-25/year
- Target: 60-100 total over 4 years (15-25/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_ichimoku_cloud_1d_tk_kijun_v1"
timeframe = "6h"
leverage = 1.0

def calculate_ichimoku_tenkan(close, period=9):
    """Tenkan-sen (Conversion Line): (highest_high + lowest_low) / 2 over period"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    result = np.zeros(n)
    for i in range(period - 1, n):
        window = close[max(0, i - period + 1):i + 1]
        result[i] = (np.max(window) + np.min(window)) / 2.0
    return result

def calculate_ichimoku_kijun(close, period=26):
    """Kijun-sen (Base Line): (highest_high + lowest_low) / 2 over period"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    result = np.zeros(n)
    for i in range(period - 1, n):
        window = close[max(0, i - period + 1):i + 1]
        result[i] = (np.max(window) + np.min(window)) / 2.0
    return result

def calculate_ichimoku_cloud(close, high, low, period=26):
    """
    Ichimoku Cloud: Senkou Span A and B
    Senkou A = (Tenkan + Kijun) / 2, plotted 26 periods ahead
    Senkou B = (highest + lowest) over 52 periods, plotted 26 ahead
    
    Cloud acts as dynamic support/resistance.
    Price ABOVE cloud = bullish, BELOW = bearish.
    """
    n = len(close)
    if n < 52:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    # Tenkan and Kijun
    tenkan = calculate_ichimoku_tenkan(close, 9)
    kijun = calculate_ichimoku_kijun(close, 26)
    
    # Senkou Span A: (Tenkan + Kijun) / 2, shifted 26 periods ahead
    senkou_a = np.full(n, np.nan)
    senkou_b = np.full(n, np.nan)
    
    for i in range(n):
        if i + 26 < n:
            if not np.isnan(tenkan[i]) and not np.isnan(kijun[i]):
                senkou_a[i + 26] = (tenkan[i] + kijun[i]) / 2.0
        
        # Senkou Span B: 52-period high/low shifted 26 ahead
        if i >= 52 - 1:
            window_high = high[max(0, i - 52 + 1):i + 1]
            window_low = low[max(0, i - 52 + 1):i + 1]
            if len(window_high) >= 52:
                senkou_b[i + 26] = (np.max(window_high) + np.min(window_low)) / 2.0
    
    return tenkan, kijun, senkou_a, senkou_b

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    CHOP > 61.8 = ranging - DON'T enter
    CHOP < 55 = trending - GOOD to enter
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.full(n, np.nan)
    for i in range(period, n):
        atr_sum = np.sum(tr[i - period + 1:i + 1])
        highest = np.max(high[i - period + 1:i + 1])
        lowest = np.min(low[i - period + 1:i + 1])
        
        if highest > lowest and atr_sum > 0:
            range_hl = highest - lowest
            chop[i] = 100 * np.log10(atr_sum / range_hl) / np.log10(period)
    
    return chop

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
    
    # === Load 1d HTF data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Ichimoku Cloud
    tenkan_1d, kijun_1d, senkou_a_1d, senkou_b_1d = calculate_ichimoku_cloud(
        df_1d['close'].values, df_1d['high'].values, df_1d['low'].values, period=26
    )
    
    # Align to 6h
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_1d)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_1d)
    tenkan_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d)
    kijun_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d)
    
    # === Local 6h indicators ===
    tenkan = calculate_ichimoku_tenkan(close, period=9)
    kijun = calculate_ichimoku_kijun(close, period=26)
    chop = calculate_choppiness(high, low, close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Volume ratio (20-period MA)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Signals
    signals = np.zeros(n)
    SIZE = 0.28  # 28% position size
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    trailing_high = 0.0
    trailing_low = 0.0
    
    warmup = 100  # 52 for cloud + 26 for kijun + 14 for chop + HTF alignment
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(tenkan[i]) or np.isnan(kijun[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        # === 1d CLOUD FILTER (HTF structure) ===
        cloud_top = max(senkou_a_aligned[i], senkou_b_aligned[i]) if not (np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i])) else np.nan
        cloud_bottom = min(senkou_a_aligned[i], senkou_b_aligned[i]) if not (np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i])) else np.nan
        
        # Price must be above cloud for long, below for short
        above_cloud = not np.isnan(cloud_top) and close[i] > cloud_top
        below_cloud = not np.isnan(cloud_bottom) and close[i] < cloud_bottom
        
        # 1d TK/Kijun cross as additional confirmation
        tenkan_above_kijun_1d = not np.isnan(tenkan_1d_aligned[i]) and not np.isnan(kijun_1d_aligned[i]) and tenkan_1d_aligned[i] > kijun_1d_aligned[i]
        tenkan_below_kijun_1d = not np.isnan(tenkan_1d_aligned[i]) and not np.isnan(kijun_1d_aligned[i]) and tenkan_1d_aligned[i] < kijun_1d_aligned[i]
        
        # === 6h TK/KIJUN CROSS (entry trigger) ===
        prev_tenkan = tenkan[i - 1]
        prev_kijun = kijun[i - 1]
        
        # Cross up: tenkan crosses above kijun
        cross_up = prev_tenkan <= prev_kijun and tenkan[i] > kijun[i]
        # Cross down: tenkan crosses below kijun
        cross_down = prev_tenkan >= prev_kijun and tenkan[i] < kijun[i]
        
        # === CHOPPINESS FILTER ===
        is_trending = chop[i] < 55
        is_choppy = chop[i] > 61.8
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.6
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG: TK cross up + above 1d cloud + 1d TK above Kijun + trending + vol spike ===
            if cross_up and above_cloud and tenkan_above_kijun_1d and is_trending and vol_spike:
                desired_signal = SIZE
            
            # === SHORT: TK cross down + below 1d cloud + 1d TK below Kijun + trending + vol spike ===
            if cross_down and below_cloud and tenkan_below_kijun_1d and is_trending and vol_spike:
                desired_signal = -SIZE
        
        # === STOPLOSS (2.5 ATR trailing stop) ===
        if in_position:
            if position_side > 0:
                # Update trailing high
                if i == entry_bar or high[i] > trailing_high:
                    trailing_high = high[i]
                
                # Trailing stop
                stop_price = trailing_high - 2.5 * entry_atr
                if low[i] < stop_price:
                    desired_signal = 0.0
                
                # Exit if cloud flips bearish
                if below_cloud:
                    desired_signal = 0.0
                
                # Exit if 1d TK/Kijun cross down
                if tenkan_below_kijun_1d:
                    desired_signal = 0.0
                
                # Exit if market becomes choppy
                if is_choppy:
                    desired_signal = 0.0
            
            elif position_side < 0:
                # Update trailing low
                if i == entry_bar or low[i] < trailing_low:
                    trailing_low = low[i]
                
                # Trailing stop
                stop_price = trailing_low + 2.5 * entry_atr
                if high[i] > stop_price:
                    desired_signal = 0.0
                
                # Exit if cloud flips bullish
                if above_cloud:
                    desired_signal = 0.0
                
                # Exit if 1d TK/Kijun cross up
                if tenkan_above_kijun_1d:
                    desired_signal = 0.0
                
                # Exit if market becomes choppy
                if is_choppy:
                    desired_signal = 0.0
        
        # === MINIMUM HOLD: 3 bars to avoid fee churn ===
        if in_position and (i - entry_bar) < 3:
            desired_signal = position_side * SIZE
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New position or flip
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                trailing_high = high[i]
                trailing_low = low[i]
        else:
            if in_position:
                in_position = False
                position_side = 0
        
        signals[i] = desired_signal
    
    return signals