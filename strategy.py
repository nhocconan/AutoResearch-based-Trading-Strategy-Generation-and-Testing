#!/usr/bin/env python3
"""
Experiment #007: 6h Ichimoku Cloud Break + Tenkan/Kijun Cross

HYPOTHESIS: Ichimoku Cloud provides institutional-level support/resistance
derived from median price calculations. Works in BOTH bull and bear because:
- Bull: price above cloud = support zones at cloud bottom
- Bear: price below cloud = resistance zones at cloud top
- Range: price inside cloud = mean reversion to cloud edges

KEY INGREDIENTS:
1. 1d Ichimoku Cloud (9/26/52 periods) for structural trend
2. 6h Tenkan/Kijun cross for entry timing
3. Choppiness filter to avoid ranging markets
4. ATR-based stoploss (2x ATR)
5. Discrete signal: 0.30

WHY 6h OVER 4h: More precise timing than 4h, fewer false signals than lower TFs.
DB pattern: CRSI/Donchian strategies work. Ichimoku is similar price-structure concept.

TARGET: 75-150 total trades over 4 years (19-37/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_ichimoku_cloud_1d_v1"
timeframe = "6h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = max(1, period // 2)
    sqrt_n = max(1, int(np.sqrt(period)))
    
    def wma(series, span):
        result = np.full(len(series), np.nan, dtype=np.float64)
        weights = np.arange(1, span + 1, dtype=np.float64)
        weight_sum = np.sum(weights)
        for i in range(span - 1, len(series)):
            if not np.isnan(series[i]):
                window = series[i - span + 1:i + 1].astype(np.float64)
                if not np.any(np.isnan(window)):
                    result[i] = np.sum(window * weights) / weight_sum
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff[i] = 2.0 * wma_half[i] - wma_full[i]
    
    return wma(diff, sqrt_n)

def calculate_ichimoku(high, low, close, tenkan=9, kijun=26, senkou_b=52):
    """
    Ichimoku Cloud calculation
    Tenkan-sen (conversion line): (highest high + lowest low) / 2 over tenkan period
    Kijun-sen (base line): (highest high + lowest low) / 2 over kijun period
    Senkou Span A (leading span A): (Tenkan + Kijun) / 2, plotted 26 periods ahead
    Senkou Span B (leading span B): (highest high + lowest low) / 2 over senkou_b, plotted 26 ahead
    Cloud = area between Senkou A and B
    """
    n = len(close)
    
    # Tenkan (fast line)
    tenkan_vals = np.full(n, np.nan, dtype=np.float64)
    for i in range(tenkan - 1, n):
        hh = np.max(high[i - tenkan + 1:i + 1])
        ll = np.min(low[i - tenkan + 1:i + 1])
        tenkan_vals[i] = (hh + ll) / 2.0
    
    # Kijun (slow line)
    kijun_vals = np.full(n, np.nan, dtype=np.float64)
    for i in range(kijun - 1, n):
        hh = np.max(high[i - kijun + 1:i + 1])
        ll = np.min(low[i - kijun + 1:i + 1])
        kijun_vals[i] = (hh + ll) / 2.0
    
    # Chikou (lagging span) - current close plotted 26 periods back
    chikou_vals = np.full(n, np.nan, dtype=np.float64)
    for i in range(kijun, n):
        chikou_vals[i - kijun] = close[i]
    
    # Senkou A (leading span A) - plotted 26 periods ahead
    senkou_a = np.full(n, np.nan, dtype=np.float64)
    for i in range(kijun - 1, n):
        if not np.isnan(tenkan_vals[i]) and not np.isnan(kijun_vals[i]):
            senkou_a[i] = (tenkan_vals[i] + kijun_vals[i]) / 2.0
    
    # Senkou B (leading span B) - plotted 26 periods ahead
    senkou_b_vals = np.full(n, np.nan, dtype=np.float64)
    for i in range(senkou_b - 1, n):
        hh = np.max(high[i - senkou_b + 1:i + 1])
        ll = np.min(low[i - senkou_b + 1:i + 1])
        senkou_b_vals[i] = (hh + ll) / 2.0
    
    return {
        'tenkan': tenkan_vals,
        'kijun': kijun_vals,
        'senkou_a': senkou_a,
        'senkou_b': senkou_b_vals,
        'chikou': chikou_vals
    }

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index - measure market choppiness"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i - period + 1:i + 1])
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and atr_sum > 0:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    
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

def calculate_donchian(high, low, period=20):
    """Donchian Channel - for breakout confirmation"""
    n = len(high)
    upper = np.full(n, np.nan, dtype=np.float64)
    lower = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load 1d data for Ichimoku cloud (HTF reference)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Ichimoku Cloud
    ichi_1d = calculate_ichimoku(
        df_1d['high'].values,
        df_1d['low'].values,
        df_1d['close'].values,
        tenkan=9, kijun=26, senkou_b=52
    )
    
    # Align 1d cloud to 6h (shift by 1 to avoid look-ahead)
    tenkan_1d_raw = ichi_1d['tenkan']
    kijun_1d_raw = ichi_1d['kijun']
    senkou_a_1d_raw = ichi_1d['senkou_a']
    senkou_b_1d_raw = ichi_1d['senkou_b']
    
    tenkan_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d_raw)
    kijun_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d_raw)
    senkou_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_1d_raw)
    senkou_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_1d_raw)
    
    # Also load 1d close for cloud top/bottom calculation
    close_1d_raw = df_1d['close'].values
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d_raw)
    
    # Calculate 6h Ichimoku for entry signals
    ichi_6h = calculate_ichimoku(high, low, close, tenkan=9, kijun=26, senkou_b=52)
    tenkan_6h = ichi_6h['tenkan']
    kijun_6h = ichi_6h['kijun']
    
    # Calculate 6h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    
    # Donchian for breakout confirmation
    dc_upper, dc_lower = calculate_donchian(high, low, period=20)
    
    # Volume analysis
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
    
    # Warmup - need senkou_b (52 periods) + kijun (26) + buffer
    warmup = 80
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(chop_14[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === ICHIMOKU CLOUD (1d) - STRUCTURAL TREND ===
        cloud_top_1d = np.nanmax([senkou_a_1d_aligned[i], senkou_b_1d_aligned[i]])
        cloud_bottom_1d = np.nanmin([senkou_a_1d_aligned[i], senkou_b_1d_aligned[i]])
        
        # 1d Tenkan/Kijun for trend
        tenkan_1d = tenkan_1d_aligned[i]
        kijun_1d = kijun_1d_aligned[i]
        
        # Check if cloud is valid
        if np.isnan(cloud_top_1d) or np.isnan(cloud_bottom_1d):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # Bullish cloud: price above cloud AND cloud is rising (senkou_a > senkou_b)
        price_above_cloud_1d = close[i] > cloud_top_1d if not np.isnan(close_1d_aligned[i]) else True
        cloud_bullish_1d = senkou_a_1d_aligned[i] > senkou_b_1d_aligned[i]
        
        # Bearish cloud: price below cloud AND cloud is falling (senkou_a < senkou_b)
        price_below_cloud_1d = close[i] < cloud_bottom_1d if not np.isnan(close_1d_aligned[i]) else False
        cloud_bearish_1d = senkou_a_1d_aligned[i] < senkou_b_1d_aligned[i]
        
        # 1d Tenkan/Kijun cross (confirms trend change)
        tenkan_above_kijun_1d = tenkan_1d > kijun_1d if not np.isnan(tenkan_1d) and not np.isnan(kijun_1d) else True
        tenkan_below_kijun_1d = tenkan_1d < kijun_1d if not np.isnan(tenkan_1d) and not np.isnan(kijun_1d) else False
        
        # === 6h TENKAN/KIJUN CROSS (ENTRY TIMING) ===
        tenkan_6h_curr = tenkan_6h[i]
        kijun_6h_curr = kijun_6h[i]
        tenkan_6h_prev = tenkan_6h[i-1] if i > 0 else np.nan
        kijun_6h_prev = kijun_6h[i-1] if i > 0 else np.nan
        
        # Bullish cross: tenkan crosses above kijun
        tk_bull_cross = False
        tk_bear_cross = False
        
        if not np.isnan(tenkan_6h_curr) and not np.isnan(kijun_6h_curr):
            if not np.isnan(tenkan_6h_prev) and not np.isnan(kijun_6h_prev):
                tk_bull_cross = (tenkan_6h_prev <= kijun_6h_prev) and (tenkan_6h_curr > kijun_6h_curr)
                tk_bear_cross = (tenkan_6h_prev >= kijun_6h_prev) and (tenkan_6h_curr < kijun_6h_curr)
        
        # === CHOPPINESS FILTER ===
        chop = chop_14[i]
        is_trending = chop < 50.0  # Lower threshold = only trending markets
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === DONCHIAN BREAKOUT ===
        dc_breakout_up = high[i] > dc_upper[i-1] if not np.isnan(dc_upper[i-1]) else False
        dc_breakout_down = low[i] < dc_lower[i-1] if not np.isnan(dc_lower[i-1]) else False
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # LONG ENTRY: 6h TK bullish cross + 1d cloud bullish + trending
        if is_trending and not in_position:
            # Bullish: price above cloud OR cloud turning bullish
            bullish_cloud = price_above_cloud_1d or (cloud_bullish_1d and tenkan_above_kijun_1d)
            
            if tk_bull_cross and bullish_cloud:
                if vol_spike or dc_breakout_up:
                    desired_signal = SIZE
                else:
                    desired_signal = SIZE  # Still enter without vol, but size matters
        
        # SHORT ENTRY: 6h TK bearish cross + 1d cloud bearish + trending
        if is_trending and not in_position:
            # Bearish: price below cloud OR cloud turning bearish
            bearish_cloud = price_below_cloud_1d or (cloud_bearish_1d and tenkan_below_kijun_1d)
            
            if tk_bear_cross and bearish_cloud:
                if vol_spike or dc_breakout_down:
                    desired_signal = -SIZE
                else:
                    desired_signal = -SIZE
        
        # === STOPLOSS CHECK (ATR-based trailing) ===
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
        
        # === TAKE PROFIT: 2.5R or cloud reversal ===
        tp_triggered = False
        if in_position and position_side > 0:
            profit_r = (close[i] - entry_price) / entry_atr
            if profit_r >= 2.5:
                tp_triggered = True
            # Also exit if cloud turns bearish
            if cloud_bearish_1d:
                tp_triggered = True
        
        if in_position and position_side < 0:
            profit_r = (entry_price - close[i]) / entry_atr
            if profit_r >= 2.5:
                tp_triggered = True
            # Also exit if cloud turns bullish
            if cloud_bullish_1d:
                tp_triggered = True
        
        if tp_triggered:
            desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
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