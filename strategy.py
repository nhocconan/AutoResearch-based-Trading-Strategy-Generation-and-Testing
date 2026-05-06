#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud with 1d trend filter and volume confirmation
# Uses 1d Ichimoku (Tenkan-sen, Kijun-sen, Senkou Span A/B, Chikou) for trend and cloud
# Price above/below cloud determines long/short bias
# Tenkan/Kijun cross provides entry signals in direction of cloud
# Volume > 1.5x 20-bar average confirms breakout strength
# Works in both bull/bear: cloud acts as dynamic support/resistance, trend filter avoids whipsaw
# Target: 60-120 total trades over 4 years (15-30/year)

name = "6h_Ichimoku_1dTrend_Filter_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 52:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Ichimoku components
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    period_tenkan = 9
    max_high_tenkan = pd.Series(high_1d).rolling(window=period_tenkan, min_periods=period_tenkan).max()
    min_low_tenkan = pd.Series(low_1d).rolling(window=period_tenkan, min_periods=period_tenkan).min()
    tenkan = (max_high_tenkan + min_low_tenkan) / 2
    
    # Kijun-sen (Base Line): (26-period high + low)/2
    period_kijun = 26
    max_high_kijun = pd.Series(high_1d).rolling(window=period_kijun, min_periods=period_kijun).max()
    min_low_kijun = pd.Series(low_1d).rolling(window=period_kijun, min_periods=period_kijun).min()
    kijun = (max_high_kijun + min_low_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + low)/2 shifted 26 periods ahead
    period_senkou_b = 52
    max_high_senkou_b = pd.Series(high_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).max()
    min_low_senkou_b = pd.Series(low_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).min()
    senkou_b = ((max_high_senkou_b + min_low_senkou_b) / 2)
    
    # Chikou Span (Lagging Span): close shifted -22 periods (but we don't use it for signals)
    
    # Calculate ATR(14) for stoploss
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate volume filter (>1.5x 20-bar average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    # Align HTF Ichimoku components to 6h timeframe
    tenkan_6h = align_htf_to_ltf(prices, df_1d, tenkan.values)
    kijun_6h = align_htf_to_ltf(prices, df_1d, kijun.values)
    senkou_a_6h = align_htf_to_ltf(prices, df_1d, senkou_a.values, additional_delay_bars=26)
    senkou_b_6h = align_htf_to_ltf(prices, df_1d, senkou_b.values, additional_delay_bars=26)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    long_stop = 0.0
    short_stop = 0.0
    
    for i in range(100, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i]) or 
            np.isnan(senkou_a_6h[i]) or np.isnan(senkou_b_6h[i]) or 
            np.isnan(atr[i]) or np.isnan(volume_filter[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
                long_stop = 0.0
                short_stop = 0.0
            continue
        
        # Determine cloud top and bottom
        cloud_top = max(senkou_a_6h[i], senkou_b_6h[i])
        cloud_bottom = min(senkou_a_6h[i], senkou_b_6h[i])
        
        if position == 0:
            # Long signal: price above cloud AND Tenkan > Kijun AND volume filter
            if close[i] > cloud_top and tenkan_6h[i] > kijun_6h[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
                long_stop = close[i] - 1.5 * atr[i]
            # Short signal: price below cloud AND Tenkan < Kijun AND volume filter
            elif close[i] < cloud_bottom and tenkan_6h[i] < kijun_6h[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
                short_stop = close[i] + 1.5 * atr[i]
        elif position == 1:
            # Update trailing stop for long
            long_stop = max(long_stop, close[i] - 1.5 * atr[i])
            # Exit long if price hits stop
            if close[i] <= long_stop:
                signals[i] = 0.0
                position = 0
                long_stop = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Update trailing stop for short
            short_stop = min(short_stop, close[i] + 1.5 * atr[i])
            # Exit short if price hits stop
            if close[i] >= short_stop:
                signals[i] = 0.0
                position = 0
                short_stop = 0.0
            else:
                signals[i] = -0.25
    
    return signals