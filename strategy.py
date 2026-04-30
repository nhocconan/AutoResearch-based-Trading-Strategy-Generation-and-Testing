#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud breakout with weekly trend filter and volume confirmation.
# Uses Ichimoku (Tenkan-sen, Kijun-sen, Senkou Span A/B) from 6h for entry signals.
# Weekly EMA200 for higher timeframe trend filter (bullish if price > weekly EMA200, bearish if <).
# Volume confirmation (>1.8x 24-bar avg) to reduce false breakouts.
# Discrete position sizing at ±0.25 to minimize fee churn.
# Target: 50-150 total trades over 4 years (12-37/year) to avoid overtrading.
# Works in bull markets via breakouts above cloud, in bear via breakdowns below cloud.
# Weekly trend filter ensures we only trade with the dominant higher timeframe momentum.

name = "6h_Ichimoku_Cloud_Breakout_WeeklyEMA200_VolumeConfirm_v1"
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
    
    # Pre-compute session hours (08-20 UTC) to avoid low liquidity periods
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 1w data ONCE before loop for weekly EMA200 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    # Calculate weekly EMA200 for trend filter
    close_1w_vals = df_1w['close'].values
    ema_200_1w = pd.Series(close_1w_vals).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Calculate Ichimoku components on 6h data
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period_tenkan = 9
    max_high_tenkan = pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    min_low_tenkan = pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan_sen = (max_high_tenkan + min_low_tenkan) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period_kijun = 26
    max_high_kijun = pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max().values
    min_low_kijun = pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun_sen = (max_high_kijun + min_low_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period_senkou_b = 52
    max_high_senkou_b = pd.Series(high).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    min_low_senkou_b = pd.Series(low).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_span_b = (max_high_senkou_b + min_low_senkou_b) / 2
    
    # Chikou Span (Lagging Span): close plotted 26 periods behind
    # Not used for entry signals to avoid look-ahead
    
    # Volume confirmation: volume > 1.8x 24-period average
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_confirm = volume > (1.8 * vol_ma_24)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(52, 24)  # warmup for Senkou Span B and volume MA
    
    for i in range(start_idx, n):
        # Skip if indicators not available or outside session
        if (np.isnan(tenkan_sen[i]) or 
            np.isnan(kijun_sen[i]) or
            np.isnan(senkou_span_a[i]) or
            np.isnan(senkou_span_b[i]) or
            np.isnan(ema_200_1w_aligned[i]) or
            np.isnan(volume_confirm[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_tenkan = tenkan_sen[i]
        curr_kijun = kijun_sen[i]
        curr_span_a = senkou_span_a[i]
        curr_span_b = senkou_span_b[i]
        curr_ema_200_1w = ema_200_1w_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        # Determine cloud top and bottom
        cloud_top = max(curr_span_a, curr_span_b)
        cloud_bottom = min(curr_span_a, curr_span_b)
        
        if position == 0:  # Flat - look for new entries
            # Bullish breakout: price above cloud, Tenkan > Kijun, above weekly EMA200, volume confirmation
            if (curr_close > cloud_top and 
                curr_tenkan > curr_kijun and 
                curr_close > curr_ema_200_1w and 
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
            # Bearish breakdown: price below cloud, Tenkan < Kijun, below weekly EMA200, volume confirmation
            elif (curr_close < cloud_bottom and 
                  curr_tenkan < curr_kijun and 
                  curr_close < curr_ema_200_1w and 
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position
            # Exit: price falls below cloud bottom or Tenkan crosses below Kijun
            if (curr_close < cloud_bottom or 
                curr_tenkan < curr_kijun):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price rises above cloud top or Tenkan crosses above Kijun
            if (curr_close > cloud_top or 
                curr_tenkan > curr_kijun):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals