#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Breakout_1dTrend_VolumeConfirm
Hypothesis: On 6h timeframe, Ichimoku cloud breakouts with 1d EMA50 trend filter and volume confirmation.
The Ichimoku cloud acts as dynamic support/resistance. TK cross (Tenkan/Kijun) provides entry timing,
while price above/below cloud confirms trend. 1d EMA50 ensures alignment with higher timeframe momentum.
Volume spike filters for institutional participation. Designed for 50-150 total trades over 4 years (12-37/year)
to stay within proven winning range for 6h timeframe. Works in bull (cloud as support in uptrend) and
bear (cloud as resistance in downtrend) markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ema(series, period):
    """Calculate Exponential Moving Average with min_periods"""
    if len(series) < period:
        return np.full_like(series, np.nan)
    return pd.Series(series).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_ichimoku(high, low, close):
    """Calculate Ichimoku components: Tenkan-sen, Kijun-sen, Senkou Span A/B, Chikou Span"""
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 plotted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 plotted 26 periods ahead
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Chikou Span (Lagging Span): close plotted 26 periods behind
    chikou = close  # Will be aligned properly in main function
    
    return tenkan, kijun, senkou_a, senkou_b, chikou

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for EMA50 trend filter (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA50 trend filter
    ema_50_1d = calculate_ema(df_1d['close'].values, 50)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Ichimoku on 6h timeframe
    tenkan, kijun, senkou_a, senkou_b, chikou = calculate_ichimoku(high, low, close)
    
    # Cloud top/bottom (Senkou Span A/B)
    cloud_top = np.maximum(senkou_a, senkou_b)
    cloud_bottom = np.minimum(senkou_a, senkou_b)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Ichimoku (52) + EMA (50) + volume MA (20)
    start_idx = max(52, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(tenkan[i]) or np.isnan(kijun[i]) or
            np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        if position == 0:
            # Look for entry signals - require: TK cross + price outside cloud + volume spike + 1d EMA50 trend alignment
            # Bullish TK cross: Tenkan crosses above Kijun
            tk_bullish_cross = tenkan[i] > kijun[i] and tenkan[i-1] <= kijun[i-1]
            # Bearish TK cross: Tenkan crosses below Kijun
            tk_bearish_cross = tenkan[i] < kijun[i] and tenkan[i-1] >= kijun[i-1]
            
            # Price above cloud (bullish) or below cloud (bearish)
            price_above_cloud = curr_close > cloud_top[i]
            price_below_cloud = curr_close < cloud_bottom[i]
            
            long_entry = (tk_bullish_cross and price_above_cloud and volume_spike[i] and 
                         (curr_close > ema_50_1d_aligned[i]))
            short_entry = (tk_bearish_cross and price_below_cloud and volume_spike[i] and 
                          (curr_close < ema_50_1d_aligned[i]))
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit when price closes back into cloud or TK cross turns bearish
            tk_bearish_cross = tenkan[i] < kijun[i] and tenkan[i-1] >= kijun[i-1]
            price_in_cloud = curr_close <= cloud_top[i] and curr_close >= cloud_bottom[i]
            
            if price_in_cloud or tk_bearish_cross:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit when price closes back into cloud or TK cross turns bullish
            tk_bullish_cross = tenkan[i] > kijun[i] and tenkan[i-1] <= kijun[i-1]
            price_in_cloud = curr_close <= cloud_top[i] and curr_close >= cloud_bottom[i]
            
            if price_in_cloud or tk_bullish_cross:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_Breakout_1dTrend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0