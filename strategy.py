#!/usr/bin/env python3

"""
Hypothesis: 6-hour Ichimoku Cloud breakout with daily trend filter and volume confirmation.
Long when price breaks above Kumo cloud and Tenkan/Kijun cross bullish with daily uptrend;
Short when price breaks below Kumo cloud and Tenkan/Kijun cross bearish with daily downtrend.
Uses volume spike for confirmation to avoid false breakouts. Designed for low trade frequency
(12-37 trades/year) by requiring multiple confirmations: Ichimoku breakout, cross, trend alignment.
Works in both bull and bear markets by following daily trend direction.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ichimoku(high, low, close, tenkan=9, kijun=26, senkou=52):
    """Calculate Ichimoku components"""
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan_sen = (pd.Series(high).rolling(window=tenkan, min_periods=tenkan).max() + 
                  pd.Series(low).rolling(window=tenkan, min_periods=tenkan).min()) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_sen = (pd.Series(high).rolling(window=kijun, min_periods=kijun).max() + 
                 pd.Series(low).rolling(window=kijun, min_periods=kijun).min()) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
    senkou_span_b = (pd.Series(high).rolling(window=senkou, min_periods=senkou).max() + 
                     pd.Series(low).rolling(window=senkou, min_periods=senkou).min()) / 2
    
    return tenkan_sen.values, kijun_sen.values, senkou_span_a.values, senkou_span_b.values

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Ichimoku on 6h
    tenkan, kijun, senkou_a, senkou_b = calculate_ichimoku(high, low, close)
    
    # Kumo cloud boundaries (shifted forward by kijun period)
    # Senkou spans are plotted kijun periods ahead
    senkou_a_shifted = np.roll(senkou_a, -kijun)
    senkou_b_shifted = np.roll(senkou_b, -kijun)
    # Fill the gap at the end with NaN
    senkou_a_shifted[-kijun:] = np.nan
    senkou_b_shifted[-kijun:] = np.nan
    
    # Kumo top and bottom
    kumo_top = np.maximum(senkou_a_shifted, senkou_b_shifted)
    kumo_bottom = np.minimum(senkou_a_shifted, senkou_b_shifted)
    
    # Daily trend filter - ONCE before loop
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 30:
        return np.zeros(n)
    
    daily_close = df_daily['close'].values
    ema34_daily = pd.Series(daily_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_daily_aligned = align_htf_to_ltf(prices, df_daily, ema34_daily)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if data not ready
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(kumo_top[i]) or np.isnan(kumo_bottom[i]) or
            np.isnan(ema34_daily_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 1.5 * vol_ma_20[i]
        
        # Ichimoku signals
        price_above_kumo = close[i] > kumo_top[i]
        price_below_kumo = close[i] < kumo_bottom[i]
        tk_cross_bullish = tenkan[i] > kijun[i] and tenkan[i-1] <= kijun[i-1]
        tk_cross_bearish = tenkan[i] < kijun[i] and tenkan[i-1] >= kijun[i-1]
        daily_uptrend = ema34_daily_aligned[i] > ema34_daily_aligned[i-1]
        daily_downtrend = ema34_daily_aligned[i] < ema34_daily_aligned[i-1]
        
        if position == 0:
            # Long: price breaks above Kumo + TK bullish cross + daily uptrend + volume spike
            if price_above_kumo and tk_cross_bullish and daily_uptrend and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Kumo + TK bearish cross + daily downtrend + volume spike
            elif price_below_kumo and tk_cross_bearish and daily_downtrend and vol_spike:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price returns to Kumo or TK cross reverses
            exit_signal = False
            
            if position == 1:
                # Exit long: price closes below Kumo or TK bearish cross
                if close[i] < kumo_top[i] or (tenkan[i] < kijun[i] and tenkan[i-1] >= kijun[i-1]):
                    exit_signal = True
            else:  # position == -1
                # Exit short: price closes above Kumo or TK bullish cross
                if close[i] > kumo_bottom[i] or (tenkan[i] > kijun[i] and tenkan[i-1] <= kijun[i-1]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "Ichimoku_Kumo_Breakout_DailyTrend_Volume"
timeframe = "6h"
leverage = 1.0