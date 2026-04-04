#!/usr/bin/env python3
"""
exp_6615_6h_ichimoku_cloud_1d_trend_v2
Hypothesis: 6h Ichimoku cloud breakout with 1d trend filter. Uses Ichimoku (Tenkan/Kijun/Senkou) on 6h for entry timing and 1d ADX for trend strength. 
Only trades in direction of 1d ADX (>25) to avoid whipsaws in ranging markets. Uses Kumo (cloud) twist as early signal. 
Discrete sizing (0.25) with ATR stoploss and max hold. Designed for 6h timeframe targeting 50-150 total trades over 4 years.
Works in bull/bear by filtering trades with higher timeframe trend strength.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_6615_6h_ichimoku_cloud_1d_trend_v2"
timeframe = "6h"
leverage = 1.0

# Parameters
ICHIMOKU_TENKAN = 9
ICHIMOKU_KIJUN = 26
ICHIMOKU_SENKOU_B = 52
ADX_PERIOD = 14
ADX_THRESHOLD = 25
VOL_MA_PERIOD = 20
VOL_BASE_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0
MAX_HOLD_BARS = 12  # ~12 * 6h = 3 days

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 1d for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d ADX for trend strength
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = pd.Series(high_1d - low_1d)
    tr2 = pd.Series(np.abs(high_1d - np.roll(close_1d, 1)))
    tr3 = pd.Series(np.abs(low_1d - np.roll(close_1d, 1)))
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr_1d.ewm(span=ADX_PERIOD, adjust=False).mean().values
    
    # Directional Movement
    dm_plus = pd.Series(np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                                 np.maximum(high_1d - np.roll(high_1d, 1), 0), 0))
    dm_minus = pd.Series(np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                                  np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0))
    
    # Smoothed DM and TR
    dm_plus_smooth = dm_plus.ewm(span=ADX_PERIOD, adjust=False).mean().values
    dm_minus_smooth = dm_minus.ewm(span=ADX_PERIOD, adjust=False).mean().values
    tr_smooth = tr_1d.ewm(span=ADX_PERIOD, adjust=False).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / tr_smooth
    di_minus = 100 * dm_minus_smooth / tr_smooth
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(span=ADX_PERIOD, adjust=False).mean().values
    
    # Align ADX to LTF (6h) with shift(1) for completed bars only
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate LTF Ichimoku components
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan = (pd.Series(high).rolling(window=ICHIMOKU_TENKAN, min_periods=ICHIMOKU_TENKAN).max().values + 
              pd.Series(low).rolling(window=ICHIMOKU_TENKAN, min_periods=ICHIMOKU_TENKAN).min().values) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun = (pd.Series(high).rolling(window=ICHIMOKU_KIJUN, min_periods=ICHIMOKU_KIJUN).max().values + 
             pd.Series(low).rolling(window=ICHIMOKU_KIJUN, min_periods=ICHIMOKU_KIJUN).min().values) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 plotted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 plotted 26 periods ahead
    senkou_b = (pd.Series(high).rolling(window=ICHIMOKU_SENKOU_B, min_periods=ICHIMOKU_SENKOU_B).max().values + 
                pd.Series(low).rolling(window=ICHIMOKU_SENKOU_B, min_periods=ICHIMOKU_SENKOU_B).min().values) / 2
    
    # Align Senkou spans to current price (they are already plotted ahead)
    senkou_a_aligned = align_htf_to_ltf(prices, prices, senkou_a)  # Using prices as dummy HTF since same TF
    senkou_b_aligned = align_htf_to_ltf(prices, prices, senkou_b)
    
    # Current Kumo (cloud) boundaries - use current Senkou spans
    upper_cloud = np.maximum(senkou_a_aligned, senkou_b_aligned)
    lower_cloud = np.minimum(senkou_a_aligned, senkou_b_aligned)
    
    # Kumo twist detection: Senkou A crossing Senkou B
    senkou_a_prev = pd.Series(senkou_a).shift(1).values
    senkou_b_prev = pd.Series(senkou_b).shift(1).values
    upper_cloud_prev = np.maximum(senkou_a_prev, senkou_b_prev)
    lower_cloud_prev = np.minimum(senkou_a_prev, senkou_b_prev)
    
    # Kumo twist bullish: Senkou A crosses above Senkou B
    kumo_twist_bullish = (senkou_a > senkou_b) & (senkou_a_prev <= senkou_b_prev)
    # Kumo twist bearish: Senkou A crosses below Senkou B
    kumo_twist_bearish = (senkou_a < senkou_b) & (senkou_a_prev >= senkou_b_prev)
    
    # Align Kumo twist signals
    kumo_twist_bullish_aligned = align_htf_to_ltf(prices, prices, kumo_twist_bullish.astype(float))
    kumo_twist_bearish_aligned = align_htf_to_ltf(prices, prices, kumo_twist_bearish.astype(float))
    
    # Price above/below cloud
    price_above_cloud = close > upper_cloud
    price_below_cloud = close < lower_cloud
    
    # Tenkan/Kijun cross
    tenkan_prev = pd.Series(tenkan).shift(1).values
    kijun_prev = pd.Series(kijun).shift(1).values
    tk_cross_bullish = (tenkan > kijun) & (tenkan_prev <= kijun_prev)
    tk_cross_bearish = (tenkan < kijun) & (tenkan_prev >= kijun_prev)
    
    # Volume MA for confirmation
    vol_ma = pd.Series(volume).rolling(window=VOL_MA_PERIOD, min_periods=VOL_MA_PERIOD).mean().values
    
    # ATR for stoploss
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = max(ICHIMOKU_SENKOU_B, ADX_PERIOD, VOL_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        bars_since_entry += 1
        
        # Skip if HTF data not available
        if (np.isnan(adx_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i]) or 
            np.isnan(upper_cloud[i]) or np.isnan(lower_cloud[i])):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= entry_price - ATR_STOP_MULTIPLIER * atr[i]:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
                continue
        elif position == -1:  # short position
            if close[i] >= entry_price + ATR_STOP_MULTIPLIER * atr[i]:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
                continue
                
        # Time-based exit
        if position != 0 and bars_since_entry >= MAX_HOLD_BARS:
            signals[i] = 0.0
            position = 0
            bars_since_entry = 0
            continue
            
        # Only trade when 1d ADX indicates strong trend (>25)
        strong_trend = adx_aligned[i] > ADX_THRESHOLD
        
        # Long conditions:
        # 1. Price above cloud (bullish bias)
        # 2. Tenkan/Kijun bullish cross OR Kumo twist bullish
        # 3. Volume confirmation
        # 4. Strong 1d trend
        long_signal = (price_above_cloud[i] and 
                      (tk_cross_bullish[i] or kumo_twist_bullish_aligned[i] > 0.5) and
                      volume[i] > vol_ma[i] * VOL_BASE_THRESHOLD and
                      strong_trend)
        
        # Short conditions:
        # 1. Price below cloud (bearish bias)
        # 2. Tenkan/Kijun bearish cross OR Kumo twist bearish
        # 3. Volume confirmation
        # 4. Strong 1d trend
        short_signal = (price_below_cloud[i] and 
                       (tk_cross_bearish[i] or kumo_twist_bearish_aligned[i] > 0.5) and
                       volume[i] > vol_ma[i] * VOL_BASE_THRESHOLD and
                       strong_trend)
        
        # Enter new positions only if flat
        if position == 0:
            if long_signal:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                bars_since_entry = 0
            elif short_signal:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                bars_since_entry = 0
            else:
                signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = position * SIGNAL_SIZE
    
    return signals