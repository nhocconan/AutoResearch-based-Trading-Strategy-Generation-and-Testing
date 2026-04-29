#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud + TK Cross + 1d ADX Trend Filter
# Long when price > Kumo (cloud), Tenkan > Kijun (TK cross bullish), and 1d ADX > 25 (strong trend)
# Short when price < Kumo, Tenkan < Kijun (TK cross bearish), and 1d ADX > 25
# Exit when TK cross reverses or price re-enters the cloud
# Uses Ichimoku for trend/momentum, ADX for regime filter to avoid whipsaws in ranging markets.
# Works in both bull and bear markets by only trading strong trends (ADX > 25).
# Discrete position sizing (0.25) to minimize fee churn.

name = "6h_Ichimoku_TK_Cross_1dADX25_Trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate ADX(14) on 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values (Wilder's smoothing)
    def WilderSmoothing(data, period):
        result = np.zeros_like(data)
        result[period-1] = np.nansum(data[:period])
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    period = 14
    atr = WilderSmoothing(tr, period)
    dm_plus_smooth = WilderSmoothing(dm_plus, period)
    dm_minus_smooth = WilderSmoothing(dm_minus, period)
    
    # DI+ and DI-
    di_plus = np.where(atr != 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr != 0, 100 * dm_minus_smooth / atr, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = WilderSmoothing(dx, period)
    
    # Align ADX to 6h timeframe (needs completed 1d bar + extra delay for smoothing)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx, additional_delay_bars=1)
    
    # Calculate Ichimoku components on 6h data
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period_tenkan = 9
    high_tenkan = pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    low_tenkan = pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (high_tenkan + low_tenkan) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period_kijun = 26
    high_kijun = pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max().values
    low_kijun = pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (high_kijun + low_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period_senkou_b = 52
    high_senkou_b = pd.Series(high).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    low_senkou_b = pd.Series(low).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = (high_senkou_b + low_senkou_b) / 2
    
    # Chikou Span (Lagging Span): close plotted 26 periods behind (not used for signals)
    
    # Kumo (Cloud) boundaries: Senkou Span A and B
    # Upper cloud: max(senkou_a, senkou_b)
    # Lower cloud: min(senkou_a, senkou_b)
    upper_cloud = np.maximum(senkou_a, senkou_b)
    lower_cloud = np.minimum(senkou_a, senkou_b)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(period_kijun, period_senkou_b, 30)  # Ensure all indicators are valid
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(upper_cloud[i]) or np.isnan(lower_cloud[i]) or 
            np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_tenkan = tenkan[i]
        curr_kijun = kijun[i]
        curr_close = close[i]
        curr_upper_cloud = upper_cloud[i]
        curr_lower_cloud = lower_cloud[i]
        curr_adx = adx_aligned[i]
        
        # TK Cross signals
        tk_bullish = curr_tenkan > curr_kijun
        tk_bearish = curr_tenkan < curr_kijun
        
        # Price relative to cloud
        price_above_cloud = curr_close > curr_upper_cloud
        price_below_cloud = curr_close < curr_lower_cloud
        price_in_cloud = (curr_close >= curr_lower_cloud) and (curr_close <= curr_upper_cloud)
        
        # Trend filter: ADX > 25 indicates strong trend
        strong_trend = curr_adx > 25
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: TK cross turns bearish OR price re-enters cloud
            if not tk_bullish or price_in_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: TK cross turns bullish OR price re-enters cloud
            if not tk_bearish or price_in_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long when: price above cloud, TK bullish, strong trend
            if price_above_cloud and tk_bullish and strong_trend:
                signals[i] = 0.25
                position = 1
            # Short when: price below cloud, TK bearish, strong trend
            elif price_below_cloud and tk_bearish and strong_trend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals