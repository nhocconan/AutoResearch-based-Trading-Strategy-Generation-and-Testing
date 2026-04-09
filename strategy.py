#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Ichimoku cloud for trend direction and 6h price action for entry timing
# Long when price is above 1d Ichimoku cloud and breaks above 6h Donchian upper (20) with volume confirmation
# Short when price is below 1d Ichimoku cloud and breaks below 6h Donchian lower (20) with volume confirmation
# Uses discrete position sizing 0.25 to target ~12-30 trades/year and minimize fee drag
# Ichimoku cloud acts as dynamic support/resistance, effective in both bull and bear markets

name = "6h_1d_ichimoku_donchian_v1"
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
    open_time = prices['open_time'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # Need at least 52 for Ichimoku calculations
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Ichimoku components on 1d
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_span_b = ((period52_high + period52_low) / 2)
    
    # Chikou Span (Lagging Span): Close shifted 26 periods behind (not used for trend)
    
    # Align Ichimoku components to 6h timeframe (no extra delay needed as Ichimoku uses completed candles)
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    
    # Calculate 6h Donchian channels (20-period)
    def rolling_max(arr, window):
        return pd.Series(arr).rolling(window=window, min_periods=window).max().values
    
    def rolling_min(arr, window):
        return pd.Series(arr).rolling(window=window, min_periods=window).min().values
    
    donchian_upper_20 = rolling_max(high, 20)
    donchian_lower_20 = rolling_min(low, 20)
    
    # Calculate 6h average volume (20-period) for confirmation
    vol_s = pd.Series(volume)
    avg_vol_20 = vol_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or
            np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i]) or
            np.isnan(donchian_upper_20[i]) or np.isnan(donchian_lower_20[i]) or
            np.isnan(avg_vol_20[i])):
            signals[i] = 0.0
            continue
        
        # Determine Ichimoku cloud boundaries and trend
        upper_cloud = np.maximum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        lower_cloud = np.minimum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        
        # Price above cloud = bullish trend, below cloud = bearish trend
        is_bullish = close[i] > upper_cloud
        is_bearish = close[i] < lower_cloud
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirmed = volume[i] > 1.5 * avg_vol_20[i]
        
        if position == 1:  # Long position
            # Exit long if price falls below Donchian lower or cloud turns bearish
            if close[i] < donchian_lower_20[i] or not is_bullish:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit short if price rises above Donchian upper or cloud turns bullish
            if close[i] > donchian_upper_20[i] or not is_bearish:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price above cloud, breaks above Donchian upper with volume
            if is_bullish and close[i] > donchian_upper_20[i] and volume_confirmed:
                position = 1
                signals[i] = 0.25
            # Enter short: price below cloud, breaks below Donchian lower with volume
            elif is_bearish and close[i] < donchian_lower_20[i] and volume_confirmed:
                position = -1
                signals[i] = -0.25
    
    return signals