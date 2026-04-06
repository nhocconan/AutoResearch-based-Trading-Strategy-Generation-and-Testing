#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud breakout with 1d trend filter and volume confirmation.
# Long when price breaks above Kumo (cloud) with bullish daily trend and volume > 1.2x 20-period average.
# Short when price breaks below Kumo with bearish daily trend and volume confirmation.
# Uses daily trend filter to avoid counter-trend trades. Ichimoku provides dynamic support/resistance.
# Target: 75-150 total trades over 4 years (19-38/year) to stay within optimal range.

name = "6h_ichimoku_1d_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Ichimoku Cloud components
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    tenkan_sen = (high_series.rolling(window=9, min_periods=9).max() + 
                  low_series.rolling(window=9, min_periods=9).min()) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    kijun_sen = (high_series.rolling(window=26, min_periods=26).max() + 
                 low_series.rolling(window=26, min_periods=26).min()) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    senkou_span_b = (high_series.rolling(window=52, min_periods=52).max() + 
                     low_series.rolling(window=52, min_periods=52).min()) / 2
    
    # Kumo (Cloud) top and bottom (shifted 26 periods ahead)
    # For backtesting, we use the cloud values that would have been known at time t
    # Senkou Span A and B are plotted 26 periods ahead, so we shift them back by 26 to align with current price
    kumo_top = np.maximum(senkou_span_a, senkou_span_b)
    kumo_bottom = np.minimum(senkou_span_a, senkou_span_b)
    
    # Daily trend filter: bullish/bearish day based on close vs open
    df_1d = get_htf_data(prices, '1d')
    daily_open = df_1d['open'].values
    daily_close = df_1d['close'].values
    daily_bullish = daily_close > daily_open  # True for bullish day
    daily_bearish = daily_close < daily_open   # True for bearish day
    daily_bullish_aligned = align_htf_to_ltf(prices, df_1d, daily_bullish)
    daily_bearish_aligned = align_htf_to_ltf(prices, df_1d, daily_bearish)
    
    # Volume filter: current volume > 1.2x 20-period average
    volume_series = pd.Series(volume)
    vol_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(52, n):  # Start after Senkou Span B calculation window
        # Skip if cloud data or daily trend data not available
        if (np.isnan(kumo_top[i]) or np.isnan(kumo_bottom[i]) or 
            np.isnan(daily_bullish_aligned[i]) or np.isnan(daily_bearish_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.2
        
        # Check exits
        if position == 1:  # long position
            # Exit: price drops below Kumo bottom or daily turn bearish
            if (close[i] < kumo_bottom[i] or 
                daily_bearish_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price rises above Kumo top or daily turn bullish
            if (close[i] > kumo_top[i] or 
                daily_bullish_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation and daily trend filter
            if volume_filter:
                # Long: break above Kumo top during bullish day
                if (close[i] > kumo_top[i] and 
                    daily_bullish_aligned[i]):
                    signals[i] = 0.25
                    position = 1
                # Short: break below Kumo bottom during bearish day
                elif (close[i] < kumo_bottom[i] and 
                      daily_bearish_aligned[i]):
                    signals[i] = -0.25
                    position = -1
    
    return signals