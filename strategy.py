#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud breakout with 1d trend filter and volume confirmation
# Uses 1d EMA50 for trend direction and Ichimoku (Tenkan/Kijun from 6h, Senkou Span from 1d) for structure
# Volume confirmation requires 1.8x average volume to ensure strong participation
# Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag on 6h timeframe
# Works in both bull and bear markets by following the 1d trend direction and using Ichimoku cloud for dynamic support/resistance
# Ichimoku is proven effective in ranging and trending markets, reducing false breakouts

name = "6h_Ichimoku_Cloud_Breakout_1dEMA50_Trend_Volume"
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
    
    # Get 6h data for Ichimoku calculation (Tenkan, Kijun)
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 52:  # Need at least 52 periods for Senkou Span B
        return np.zeros(n)
    
    # Get 1d data for trend filter and Senkou Span
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Ichimoku components on 6h: Tenkan-sen (9-period), Kijun-sen (26-period)
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Tenkan-sen: (9-period high + 9-period low) / 2
    tenkan_period = 9
    tenkan_sen = (pd.Series(high_6h).rolling(window=tenkan_period, min_periods=tenkan_period).max() + 
                  pd.Series(low_6h).rolling(window=tenkan_period, min_periods=tenkan_period).min()) / 2
    tenkan_sen = tenkan_sen.values
    
    # Kijun-sen: (26-period high + 26-period low) / 2
    kijun_period = 26
    kijun_sen = (pd.Series(high_6h).rolling(window=kijun_period, min_periods=kijun_period).max() + 
                 pd.Series(low_6h).rolling(window=kijun_period, min_periods=kijun_period).min()) / 2
    kijun_sen = kijun_sen.values
    
    # Senkou Span A: (Tenkan-sen + Kijun-sen) / 2, plotted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B: (52-period high + 52-period low) / 2, plotted 26 periods ahead
    senkou_b_period = 52
    senkou_span_b = (pd.Series(high_6h).rolling(window=senkou_b_period, min_periods=senkou_b_period).max() + 
                     pd.Series(low_6h).rolling(window=senkou_b_period, min_periods=senkou_b_period).min()) / 2
    senkou_span_b = senkou_span_b.values
    
    # Align Ichimoku components to 6h timeframe with proper look-ahead prevention
    # Tenkan and Kijun are aligned normally (they use completed 6h bars)
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_6h, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_6h, kijun_sen)
    # Senkou Span needs extra 26-bar delay because it's plotted 26 periods ahead
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_6h, senkou_span_a, additional_delay_bars=26)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_6h, senkou_span_b, additional_delay_bars=26)
    
    # Volume confirmation: 20-period EMA on 6h volume
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start from 100 to have valid indicators
        # Skip if any value is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(tenkan_sen_aligned[i]) or 
            np.isnan(kijun_sen_aligned[i]) or np.isnan(senkou_span_a_aligned[i]) or 
            np.isnan(senkou_span_b_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 1.8 x 20-period EMA (balanced to avoid overtrading)
        volume_spike = volume[i] > (1.8 * vol_ema_20[i])
        
        # Ichimoku Cloud: Top = max(Senkou Span A, Senkou Span B), Bottom = min(Senkou Span A, Senkou Span B)
        cloud_top = np.maximum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        cloud_bottom = np.minimum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        
        # Tenkan-Kijun cross: bullish when Tenkan > Kijun, bearish when Tenkan < Kijun
        tk_bullish = tenkan_sen_aligned[i] > kijun_sen_aligned[i]
        tk_bearish = tenkan_sen_aligned[i] < kijun_sen_aligned[i]
        
        # Ichimoku signal conditions:
        # Long: Price breaks above cloud + volume spike + TK bullish + price above 1d EMA50 (uptrend)
        # Short: Price breaks below cloud + volume spike + TK bearish + price below 1d EMA50 (downtrend)
        if position == 0:
            if (close[i] > cloud_top and volume_spike and tk_bullish and 
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            elif (close[i] < cloud_bottom and volume_spike and tk_bearish and 
                  close[i] < ema_50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price breaks below cloud bottom OR TK bearish cross OR price below 1d EMA50 (trend change)
            if (close[i] < cloud_bottom or not tk_bullish or 
                close[i] < ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price breaks above cloud top OR TK bullish cross OR price above 1d EMA50 (trend change)
            if (close[i] > cloud_top or not tk_bearish or 
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals