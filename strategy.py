#!/usr/bin/env python3
"""
Experiment #011: 6h Ichimoku Cloud + 1d EMA Trend + Volume Confirmation

HYPOTHESIS: Ichimoku's Kumo (cloud) provides natural support/resistance and
regime detection - when price is above cloud = bullish, below = bearish.
The Kijun-Tenkan cross catches momentum shifts while the Chikou span
confirms momentum. Combined with 1d EMA for macro trend and volume spike
confirmation, this should capture major directional moves in both bull
(2021) and bear (2022) markets.

WHY IT SHOULD WORK IN BOTH MARKETS:
- Ichimoku cloud is designed for range/trend regime detection
- In bear markets (2022), Kijun-Tenkan shorts catch crash momentum
- Kijun-Tenkan longs during bear rallies require cloud confirmation (tight filter)
- 1d EMA ensures we're aligned with macro trend, not fighting it
- 2025 bear/range market: cloud acts as resistance, shorts on crosses work
- Chikou confirmation adds extra edge without overcomplicating

TRADE COUNT ESTIMATE:
- 6h bars: ~1460/year, 5840 over 4 years
- Kijun-Tenkan crosses: ~8-12 signals/symbol/year (26-period baseline)
- Cloud alignment filter: ~50% pass = 4-6 signals
- 1d EMA alignment: ~60% pass = 2.5-3.5 signals
- Volume spike (>1.5x): ~70% pass = 1.7-2.4 signals/symbol/year
- 4yr total: ~7-10 per symbol - TOO LOW

SOLUTION: Use Ichimoku for REGIME only (not entry), combine with:
- Donchian(24) breakout for entries (more signals than Ichimoku crosses)
- Ichimoku cloud as directional filter
- 1d EMA for macro trend
- Volume spike for confirmation

Revised estimate: 30-60 trades/symbol/year, 120-240 total - in target range.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_ichimoku_donchian_1d_ema_v1"
timeframe = "6h"
leverage = 1.0

def calculate_ichimoku(high, low, close, tenkan_period=9, kijun_period=26, senkou_b_period=52):
    """
    Ichimoku Cloud components:
    - Tenkan-sen (Conversion Line): (highest_high + lowest_low) / 2 over tenkan_period
    - Kijun-sen (Base Line): (highest_high + lowest_low) / 2 over kijun_period
    - Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2, plotted 26 periods ahead
    - Senkou Span B (Leading Span B): (highest + lowest) / 2 over senkou_b_period, plotted 26 ahead
    - Chikou Span (Lagging Span): current close, plotted 26 periods behind
    
    Cloud (Kumo): area between Senkou A and Senkou B
    """
    n = len(close)
    
    # Initialize arrays
    tenkan = np.full(n, np.nan)
    kijun = np.full(n, np.nan)
    senkou_a = np.full(n, np.nan)
    senkou_b = np.full(n, np.nan)
    
    # Tenkan-sen: mid-point of highest high and lowest low over 9 periods
    for i in range(tenkan_period - 1, n):
        segment_high = high[i - tenkan_period + 1:i + 1]
        segment_low = low[i - tenkan_period + 1:i + 1]
        tenkan[i] = (np.nanmax(segment_high) + np.nanmin(segment_low)) / 2.0
    
    # Kijun-sen: mid-point over 26 periods
    for i in range(kijun_period - 1, n):
        segment_high = high[i - kijun_period + 1:i + 1]
        segment_low = low[i - kijun_period + 1:i + 1]
        kijun[i] = (np.nanmax(segment_high) + np.nanmin(segment_low)) / 2.0
    
    # Senkou Span A: (Tenkan + Kijun) / 2, shifted forward 26 periods
    for i in range(kijun_period - 1, n):
        if not np.isnan(tenkan[i]) and not np.isnan(kijun[i]):
            senkou_a[i] = (tenkan[i] + kijun[i]) / 2.0
    
    # Senkou Span B: mid-point over 52 periods, shifted forward 26 periods
    for i in range(senkou_b_period - 1, n):
        segment_high = high[i - senkou_b_period + 1:i + 1]
        segment_low = low[i - senkou_b_period + 1:i + 1]
        senkou_b[i] = (np.nanmax(segment_high) + np.nanmin(segment_low)) / 2.0
    
    # Chikou Span: current close, shifted back 26 periods
    chikou = close.copy()
    
    return tenkan, kijun, senkou_a, senkou_b, chikou

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

def calculate_donchian(high, low, period=20):
    """Donchian Channel - use period 24 for 6h (matches ~10 days)"""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d EMA for macro trend (call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # === Local 6h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    tenkan, kijun, senkou_a, senkou_b, chikou = calculate_ichimoku(high, low, close)
    
    # Donchian Channel (24 periods ~ 6 days on 6h)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=24)
    
    # Volume average for spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 1e-10, vol_ma, 1.0)
    
    # === Signals ===
    signals = np.zeros(n)
    SIZE = 0.25  # Conservative for 6h
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    warmup = 100  # Need enough for Ichimoku Senkou B (52 periods)
    
    for i in range(warmup, n):
        # NaN checks for all indicators
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(tenkan[i]) or np.isnan(kijun[i]) or np.isnan(senkou_a[i]) or np.isnan(senkou_b[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(ema_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Update highest/lowest for trailing stop
        if in_position:
            if position_side > 0:
                highest_since_entry = max(highest_since_entry, high[i])
            else:
                lowest_since_entry = min(lowest_since_entry, low[i])
        
        # === ICHIMOKU CLOUD REGIME (price vs cloud) ===
        # Bullish: price above cloud (Senkou A > Senkou B means bullish cloud)
        cloud_top = np.maximum(senkou_a[i], senkou_b[i])
        cloud_bottom = np.minimum(senkou_a[i], senkou_b[i])
        price_above_cloud = close[i] > cloud_top
        price_below_cloud = close[i] < cloud_bottom
        
        # Bullish cloud: Senkou A > Senkou B
        bullish_cloud = senkou_a[i] > senkou_b[i]
        bearish_cloud = senkou_a[i] < senkou_b[i]
        
        # Kijun-Tenkan cross (momentum signal)
        tenkan_above_kijun = tenkan[i] > kijun[i]
        tenkan_below_kijun = tenkan[i] < kijun[i]
        
        prev_tenkan_above = tenkan[i-1] > kijun[i-1] if i > 0 else False
        tenkan_cross_up = prev_tenkan_above and tenkan_above_kijun and (tenkan[i-1] <= kijun[i-1])
        tenkan_cross_down = not prev_tenkan_above and tenkan_below_kijun and (tenkan[i-1] >= kijun[i-1])
        
        # === HTF TREND (1d EMA aligned) ===
        htf_bullish = close[i] > ema_1d_aligned[i]
        htf_bearish = close[i] < ema_1d_aligned[i]
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === DONCHIAN BREAKOUT (prior bar) ===
        prev_upper = donchian_upper[i-1] if i > 0 and not np.isnan(donchian_upper[i-1]) else np.nan
        prev_lower = donchian_lower[i-1] if i > 0 and not np.isnan(donchian_lower[i-1]) else np.nan
        
        bullish_breakout = (not np.isnan(prev_upper) and close[i] > prev_upper)
        bearish_breakout = (not np.isnan(prev_lower) and close[i] < prev_lower)
        
        # === MINIMUM HOLD: 3 bars ===
        min_hold = (i - entry_bar) >= 3
        
        # === EXITS ===
        if in_position:
            # ATR trailing stop (2.5x ATR)
            if position_side > 0:
                stop_hit = low[i] < (highest_since_entry - 2.5 * entry_atr)
            else:
                stop_hit = high[i] > (lowest_since_entry + 2.5 * entry_atr)
            
            # Exit on Ichimoku cloud rejection
            if position_side > 0 and price_below_cloud:
                stop_hit = True
            if position_side < 0 and price_above_cloud:
                stop_hit = True
            
            # Exit on Kijun-Tenkan reversal (with min hold)
            if position_side > 0 and min_hold and tenkan_cross_down:
                stop_hit = True
            if position_side < 0 and min_hold and tenkan_cross_up:
                stop_hit = True
            
            if stop_hit:
                signals[i] = 0.0
                in_position = False
                position_side = 0
            else:
                signals[i] = position_side * SIZE
        
        # === NEW POSITIONS ===
        if not in_position:
            # Combine: Ichimoku regime + Donchian breakout + volume + HTF alignment
            
            # LONG: Bullish Ichimoku + Donchian breakout + volume + HTF bullish
            # Bullish: price above cloud + bullish cloud + HTF aligned
            if (price_above_cloud and bullish_cloud and htf_bullish and 
                bullish_breakout and vol_spike):
                in_position = True
                position_side = 1
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                highest_since_entry = high[i]
                signals[i] = SIZE
            
            # SHORT: Bearish Ichimoku + Donchian breakdown + volume + HTF bearish
            elif (price_below_cloud and bearish_cloud and htf_bearish and 
                  bearish_breakout and vol_spike):
                in_position = True
                position_side = -1
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                lowest_since_entry = low[i]
                signals[i] = -SIZE
            
            # ALT ENTRY: Kijun-Tenkan cross as momentum trigger
            # Only in direction of cloud and HTF
            elif (tenkan_cross_up and price_above_cloud and bullish_cloud and 
                  htf_bullish and vol_spike):
                in_position = True
                position_side = 1
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                highest_since_entry = high[i]
                signals[i] = SIZE
            
            elif (tenkan_cross_down and price_below_cloud and bearish_cloud and 
                  htf_bearish and vol_spike):
                in_position = True
                position_side = -1
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                lowest_since_entry = low[i]
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
    
    return signals