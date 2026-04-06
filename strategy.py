#!/usr/bin/env python3
"""
6H Ichimoku Cloud Filter with TK Cross and Weekly Pivot Direction
Hypothesis: Ichimoku cloud acts as dynamic support/resistance, TK cross provides momentum signals,
and weekly pivot direction filters for higher-probability trades. Works in bull/bear by using
cloud as trend filter and avoiding counter-trend signals. Targets 50-150 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ichimoku_tk_weekly_pivot_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load weekly data for pivot direction (once before loop)
    df_weekly = get_htf_data(prices, '1w')
    close_weekly = df_weekly['close'].values
    
    # Weekly pivot points (using prior week's OHLC)
    pivot = np.full(n, np.nan)
    r1 = np.full(n, np.nan)
    s1 = np.full(n, np.nan)
    r2 = np.full(n, np.nan)
    s2 = np.full(n, np.nan)
    r3 = np.full(n, np.nan)
    s3 = np.full(n, np.nan)
    
    # Calculate weekly pivots (shifted by 1 to avoid look-ahead)
    for i in range(len(close_weekly)):
        if i >= 1:  # Need prior week's data
            # Use prior week's high, low, close (already completed)
            idx = i - 1
            # Approximate weekly OHLC from daily - we'll use weekly high/low/close from df_weekly
            # Since we have weekly data, we need to get the actual weekly OHLC
            # For simplicity, we'll use close as proxy and calculate proper pivot later
            pass
    
    # Better approach: calculate pivot from weekly OHLC
    # Extract weekly OHLC from the weekly dataframe
    if 'high' in df_weekly.columns and 'low' in df_weekly.columns:
        weekly_high = df_weekly['high'].values
        weekly_low = df_weekly['low'].values
        weekly_close = df_weekly['close'].values
        weekly_open = df_weekly['open'].values if 'open' in df_weekly.columns else weekly_close
        
        # Calculate pivot points for each week
        pivot_weekly = (weekly_high + weekly_low + weekly_close) / 3
        r1_weekly = 2 * pivot_weekly - weekly_low
        s1_weekly = 2 * pivot_weekly - weekly_high
        r2_weekly = pivot_weekly + (weekly_high - weekly_low)
        s2_weekly = pivot_weekly - (weekly_high - weekly_low)
        r3_weekly = weekly_high + 2 * (pivot_weekly - weekly_low)
        s3_weekly = weekly_low - 2 * (weekly_high - pivot_weekly)
        
        # Align to 6h timeframe
        pivot = align_htf_to_ltf(prices, df_weekly, pivot_weekly)
        r1 = align_htf_to_ltf(prices, df_weekly, r1_weekly)
        s1 = align_htf_to_ltf(prices, df_weekly, s1_weekly)
        r2 = align_htf_to_ltf(prices, df_weekly, r2_weekly)
        s2 = align_htf_to_ltf(prices, df_weekly, s2_weekly)
        r3 = align_htf_to_ltf(prices, df_weekly, r3_weekly)
        s3 = align_htf_to_ltf(prices, df_weekly, s3_weekly)
    else:
        # Fallback: use close only (less accurate but functional)
        pivot = close_weekly
        r1 = close_weekly * 1.01
        s1 = close_weekly * 0.99
        r2 = close_weekly * 1.02
        s2 = close_weekly * 0.98
        r3 = close_weekly * 1.03
        s3 = close_weekly * 0.97
        pivot = align_htf_to_ltf(prices, df_weekly, pivot)
        r1 = align_htf_to_ltf(prices, df_weekly, r1)
        s1 = align_htf_to_ltf(prices, df_weekly, s1)
        r2 = align_htf_to_ltf(prices, df_weekly, r2)
        s2 = align_htf_to_ltf(prices, df_weekly, s2)
        r3 = align_htf_to_ltf(prices, df_weekly, r3)
        s3 = align_htf_to_ltf(prices, df_weekly, s3)
    
    # 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9 = 9
    max_high_9 = np.full(n, np.nan)
    min_low_9 = np.full(n, np.nan)
    for i in range(period9, n):
        max_high_9[i] = np.max(high[i-period9:i])
        min_low_9[i] = np.min(low[i-period9:i])
    tenkan = (max_high_9 + min_low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26 = 26
    max_high_26 = np.full(n, np.nan)
    min_low_26 = np.full(n, np.nan)
    for i in range(period26, n):
        max_high_26[i] = np.max(high[i-period26:i])
        min_low_26[i] = np.min(low[i-period26:i])
    kijun = (max_high_26 + min_low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    # We'll handle the shift in the logic by using current values for cloud
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 52 periods ahead
    period52 = 52
    max_high_52 = np.full(n, np.nan)
    min_low_52 = np.full(n, np.nan)
    for i in range(period52, n):
        max_high_52[i] = np.max(high[i-period52:i])
        min_low_52[i] = np.min(low[i-period52:i])
    senkou_b = (max_high_52 + min_low_52) / 2
    
    # Chikou Span (Lagging Span): Close shifted 26 periods back
    # Not used in this strategy to avoid look-ahead
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from warmup period
    start = max(period52, 26)  # Need Senkou B calculated
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(senkou_a[i]) or np.isnan(senkou_b[i]) or
            np.isnan(pivot[i]) or np.isnan(r1[i]) or np.isnan(s1[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Determine cloud color and position
        # Cloud top = max(senkou_a, senkou_b), Cloud bottom = min(senkou_a, senkou_b)
        cloud_top = max(senkou_a[i], senkou_b[i])
        cloud_bottom = min(senkou_a[i], senkou_b[i])
        
        # TK Cross
        tk_cross = tenkan[i] > kijun[i]  # Bullish when Tenkan > Kijun
        
        # Price relative to cloud
        price_above_cloud = close[i] > cloud_top
        price_below_cloud = close[i] < cloud_bottom
        price_in_cloud = (close[i] >= cloud_bottom) and (close[i] <= cloud_top)
        
        # Weekly pivot direction (bullish if price above weekly pivot)
        bullish_pivot = close[i] > pivot[i]
        bearish_pivot = close[i] < pivot[i]
        
        # Volume filter: above average volume
        vol_ma = np.full(n, np.nan)
        if i >= 20:
            vol_ma[i] = np.mean volume[i-20:i] if hasattr(np, 'mean') else np.mean(volume[i-20:i])
        else:
            vol_ma[i] = np.mean(volume[:i+1]) if i > 0 else volume[i]
        volume_filter = volume[i] > vol_ma[i] * 1.2 if not np.isnan(vol_ma[i]) else True
        
        # Entry logic
        if position == 0:  # Look for new entries
            # Long: Price above cloud, TK cross bullish, and bullish weekly pivot
            if price_above_cloud and tk_cross and bullish_pivot and volume_filter:
                signals[i] = 0.25
                position = 1
            # Short: Price below cloud, TK cross bearish, and bearish weekly pivot
            elif price_below_cloud and not tk_cross and bearish_pivot and volume_filter:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long position - exit conditions
            # Exit: Price falls below cloud OR TK cross turns bearish
            if not price_above_cloud or not tk_cross:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position - exit conditions
            # Exit: Price rises above cloud OR TK cross turns bullish
            if not price_below_cloud or tk_cross:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    # Handle volume calculation properly
    # Recalculate volume MA with proper loop
    vol_ma = np.full(n, np.nan)
    for i in range(n):
        if i >= 20:
            vol_ma[i] = np.mean(volume[i-20:i])
        elif i > 0:
            vol_ma[i] = np.mean(volume[:i])
        else:
            vol_ma[i] = volume[i]
    
    # Re-run with correct volume MA (simplified - in practice we'd compute once)
    # For now, return the signals as computed (volume filter logic needs fix)
    # Let's simplify and recompute properly
    
    # Recompute with proper volume MA
    signals = np.zeros(n)
    position = 0
    
    # Precompute volume MA
    vol_ma = np.full(n, np.nan)
    for i in range(n):
        if i >= 20:
            vol_ma[i] = np.mean(volume[i-20:i])
        elif i > 0:
            vol_ma[i] = np.mean(volume[:i])
        else:
            vol_ma[i] = volume[i] if n > 0 else 0
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(senkou_a[i]) or np.isnan(senkou_b[i]) or
            np.isnan(pivot[i]) or np.isnan(r1[i]) or np.isnan(s1[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Determine cloud color and position
        cloud_top = max(senkou_a[i], senkou_b[i])
        cloud_bottom = min(senkou_a[i], senkou_b[i])
        
        # TK Cross
        tk_cross = tenkan[i] > kijun[i]
        
        # Price relative to cloud
        price_above_cloud = close[i] > cloud_top
        price_below_cloud = close[i] < cloud_bottom
        
        # Weekly pivot direction
        bullish_pivot = close[i] > pivot[i]
        bearish_pivot = close[i] < pivot[i]
        
        # Volume filter
        volume_filter = volume[i] > vol_ma[i] * 1.5
        
        # Entry logic
        if position == 0:  # Look for new entries
            # Long: Price above cloud, TK cross bullish, and bullish weekly pivot
            if price_above_cloud and tk_cross and bullish_pivot and volume_filter:
                signals[i] = 0.25
                position = 1
            # Short: Price below cloud, TK cross bearish, and bearish weekly pivot
            elif price_below_cloud and not tk_cross and bearish_pivot and volume_filter:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long position - exit conditions
            # Exit: Price falls below cloud OR TK cross turns bearish
            if not price_above_cloud or not tk_cross:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position - exit conditions
            # Exit: Price rises above cloud OR TK cross turns bullish
            if not price_below_cloud or tk_cross:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals