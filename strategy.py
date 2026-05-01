#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud + 1d Volume Spike + 1w Pivot Direction
# Uses 1w Camarilla pivot for structural regime (bull/bear/range).
# Long when: Price above Ichimoku cloud (Senkou Span A & B) AND TK cross bullish (Tenkan > Kijun) AND 1d volume > 1.5x 20-period average AND price above 1w pivot.
# Short when: Price below Ichimoku cloud AND TK cross bearish AND 1d volume spike AND price below 1w pivot.
# Ichimoku provides trend, support/resistance, and momentum in one indicator.
# Volume spike confirms institutional participation.
# 1w pivot avoids counter-trend trades in strong regimes.
# Works in bull (trend with cloud) and bear (counter-trend at extremes via pivot).
# Target: 15-30 trades/year.

name = "6h_Ichimoku_Cloud_VolumeSpike_1wPivot_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours for efficiency
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # Load 1d data ONCE for Ichimoku and volume average
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Load 1w data ONCE for pivot direction
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Ichimoku components (using 1d data)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period_tenkan = 9
    max_high_tenkan = pd.Series(high_1d).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    min_low_tenkan = pd.Series(low_1d).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (max_high_tenkan + min_low_tenkan) / 2.0
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period_kijun = 26
    max_high_kijun = pd.Series(high_1d).rolling(window=period_kijun, min_periods=period_kijun).max().values
    min_low_kijun = pd.Series(low_1d).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (max_high_kijun + min_low_kijun) / 2.0
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2, shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2.0)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2, shifted 26 periods ahead
    period_senkou_b = 52
    max_high_senkou_b = pd.Series(high_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    min_low_senkou_b = pd.Series(low_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = ((max_high_senkou_b + min_low_senkou_b) / 2.0)
    
    # Chikou Span (Lagging Span): Close plotted 26 periods behind
    # Not used in signals as it's lagging
    
    # Align Ichimoku components to 6h
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # 1d Volume Spike: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    # 1w Camarilla pivot points (using prior week OHLC)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Prior week OHLC for current week's pivot
    prev_high = np.roll(high_1w, 1)
    prev_low = np.roll(low_1w, 1)
    prev_close = np.roll(close_1w, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    pivot_1w = (prev_high + prev_low + prev_close) / 3.0
    range_1w = prev_high - prev_low
    r1_1w = prev_close + (range_1w * 1.1 / 12)
    s1_1w = prev_close - (range_1w * 1.1 / 12)
    
    # Align 1w levels to 6h
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # warmup for Ichimoku (max 52-period + 26 shift)
    
    for i in range(start_idx, n):
        # Session filter: 08-20 UTC (reduce noise, focus on active sessions)
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            # Outside session: flatten position if any
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Skip if any data not ready
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or
            np.isnan(pivot_1w_aligned[i]) or np.isnan(r1_1w_aligned[i]) or
            np.isnan(s1_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_tenkan = tenkan_aligned[i]
        curr_kijun = kijun_aligned[i]
        curr_senkou_a = senkou_a_aligned[i]
        curr_senkou_b = senkou_b_aligned[i]
        curr_pivot = pivot_1w_aligned[i]
        curr_r1 = r1_1w_aligned[i]
        curr_s1 = s1_1w_aligned[i]
        curr_volume_spike = volume_spike[i]
        
        # Ichimoku cloud: Senkou Span A and B form the cloud
        # Cloud top = max(Senkou A, Senkou B)
        # Cloud bottom = min(Senkou A, Senkou B)
        cloud_top = max(curr_senkou_a, curr_senkou_b)
        cloud_bottom = min(curr_senkou_a, curr_senkou_b)
        
        # Price above/below cloud
        price_above_cloud = curr_close > cloud_top
        price_below_cloud = curr_close < cloud_bottom
        
        # TK cross: Tenkan > Kijun (bullish), Tenkan < Kijun (bearish)
        tk_bullish = curr_tenkan > curr_kijun
        tk_bearish = curr_tenkan < curr_kijun
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: Price above cloud AND TK cross bullish AND volume spike AND price above weekly pivot
            if (price_above_cloud and 
                tk_bullish and 
                curr_volume_spike and 
                curr_close > curr_pivot):
                signals[i] = 0.25
                position = 1
            # Short: Price below cloud AND TK cross bearish AND volume spike AND price below weekly pivot
            elif (price_below_cloud and 
                  tk_bearish and 
                  curr_volume_spike and 
                  curr_close < curr_pivot):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price breaks below cloud OR TK cross bearish OR price breaks below weekly S1
            if (not price_above_cloud or 
                not tk_bullish or 
                curr_close < curr_s1):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Price breaks above cloud OR TK cross bullish OR price breaks above weekly R1
            if (not price_below_cloud or 
                not tk_bearish or 
                curr_close > curr_r1):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals