#!/usr/bin/env python3
"""
Hypothesis: 6h Ichimoku Cloud breakout with 1w trend filter and volume confirmation.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1w for EMA200 trend direction (bull/bear filter) and 1d for volume spike confirmation.
- Ichimoku Components (6h):
  * Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
  * Kijun-sen (Base Line): (26-period high + 26-period low)/2
  * Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2, plotted 26 periods ahead
  * Senkou Span B (Leading Span B): (52-period high + 52-period low)/2, plotted 26 periods ahead
  * Kumo (Cloud): Between Senkou Span A and B
- Entry: Long when price breaks above Kumo AND Tenkan > Kijun (bullish TK cross) AND 1w EMA200 uptrend AND volume > 2.0 * 20-period average volume.
         Short when price breaks below Kumo AND Tenkan < Kijun (bearish TK cross) AND 1w EMA200 downtrend AND volume > 2.0 * 20-period average volume.
- Exit: Opposite Ichimoku signal (long exits when price breaks below Kumo, short exits when price breaks above Kumo).
- Signal size: 0.25 discrete to minimize fee drag.
- Works in both bull and bear markets by using 1w EMA200 to filter trend direction and Ichimoku for dynamic support/resistance.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:  # Need sufficient data for Ichimoku calculations
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1w EMA200 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:  # Need sufficient data for EMA200
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # Calculate 1d volume average for confirmation (20-period)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    vol_ma_20_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Calculate Ichimoku components on 6h timeframe
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period_tenkan = 9
    highest_high_9 = pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    lowest_low_9 = pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan_sen = (highest_high_9 + lowest_low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period_kijun = 26
    highest_high_26 = pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max().values
    lowest_low_26 = pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun_sen = (highest_high_26 + lowest_low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
    period_senkou_b = 52
    highest_high_52 = pd.Series(high).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    lowest_low_52 = pd.Series(low).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_span_b = (highest_high_52 + lowest_low_52) / 2
    
    # Determine cloud boundaries (Senkou Span A and B shifted 26 periods ahead)
    # For current price, we use the cloud values that were plotted 26 periods ago
    displacement = 26
    if len(senkou_span_a) > displacement:
        # Cloud top/bottom for current index i comes from senkou values at i-displacement
        senkou_span_a_lagged = np.roll(senkou_span_a, displacement)
        senkou_span_b_lagged = np.roll(senkou_span_b, displacement)
        # First 'displacement' values are invalid (rolled from end)
        senkou_span_a_lagged[:displacement] = np.nan
        senkou_span_b_lagged[:displacement] = np.nan
    else:
        senkou_span_a_lagged = np.full_like(senkou_span_a, np.nan)
        senkou_span_b_lagged = np.full_like(senkou_span_b, np.nan)
    
    # Cloud top is the higher of Senkou Span A and B
    # Cloud bottom is the lower of Senkou Span A and B
    cloud_top = np.maximum(senkou_span_a_lagged, senkou_span_b_lagged)
    cloud_bottom = np.minimum(senkou_span_a_lagged, senkou_span_b_lagged)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    # Need: 52 for Senkou B, 26 for displacement, 200 for 1w EMA, 20 for 1d volume MA
    start_idx = max(52 + 26, 200, 20)  # 78, 200, 20 -> 200
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema200_1w_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i]) or
            np.isnan(tenkan_sen[i]) or np.isnan(kijun_sen[i]) or
            np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        curr_tenkan = tenkan_sen[i]
        curr_kijun = kijun_sen[i]
        curr_cloud_top = cloud_top[i]
        curr_cloud_bottom = cloud_bottom[i]
        
        # Trend filter: 1w EMA200 direction
        long_term_uptrend = curr_close > ema200_1w_aligned[i]
        long_term_downtrend = curr_close < ema200_1w_aligned[i]
        
        # Volume confirmation: current volume > 2.0 * 20-period average volume
        volume_confirm = curr_volume > 2.0 * vol_ma_20_1d_aligned[i] if not np.isnan(vol_ma_20_1d_aligned[i]) else False
        
        # Ichimoku signals
        price_above_cloud = curr_close > curr_cloud_top
        price_below_cloud = curr_close < curr_cloud_bottom
        tk_bullish = curr_tenkan > curr_kijun  # Tenkan above Kijun
        tk_bearish = curr_tenkan < curr_kijun  # Tenkan below Kijun
        
        # Exit conditions: opposite Ichimoku signal
        if position != 0:
            # Exit long: price breaks below cloud (bearish Ichimoku exit)
            if position == 1:
                if price_below_cloud:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price breaks above cloud (bullish Ichimoku exit)
            elif position == -1:
                if price_above_cloud:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Ichimoku breakout with trend and volume filters
        if position == 0:
            # Long: price breaks above cloud AND bullish TK cross AND long-term uptrend AND volume confirmation
            long_condition = price_above_cloud and tk_bullish and long_term_uptrend and volume_confirm
            
            # Short: price breaks below cloud AND bearish TK cross AND long-term downtrend AND volume confirmation
            short_condition = price_below_cloud and tk_bearish and long_term_downtrend and volume_confirm
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_1wEMA200Trend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0