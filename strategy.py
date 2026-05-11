#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Filter_1dTrend
Hypothesis: Ichimoku cloud from 1d provides strong support/resistance. On 6h, enter long when Tenkan-sen > Kijun-sen and price above cloud (bullish), short when Tenkan-sen < Kijun-sen and price below cloud (bearish). Use 1d ADX > 25 to confirm trend regime and avoid whipsaws in ranging markets. Volume confirmation (6h volume > 1.5x 20-period average) filters weak signals. Designed for 6h timeframe to target 50-150 total trades over 4 years.
"""

name = "6h_Ichimoku_Cloud_Filter_1dTrend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 1d data for Ichimoku and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # 6h OHLCV
    close_6h = prices['close'].values
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    volume_6h = prices['volume'].values
    
    # --- 1d Ichimoku Cloud ---
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_span_b = (period52_high + period52_low) / 2
    
    # Chikou Span (Lagging Span): not used for signals
    
    # Align Ichimoku components to 6h
    tenkan_sen_6h = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_6h = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_span_a_6h = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_6h = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    
    # --- 1d ADX for trend filter (14 period) ---
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Directional Movement
    up_move = np.diff(high_1d, prepend=high_1d[0])
    down_move = -np.diff(low_1d, prepend=low_1d[0])
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed DM
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr_1d
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr_1d
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    adx_6h_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # --- 6h Volume Average for confirmation ---
    vol_avg_6h = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start after warmup period for Ichomoku (52 periods) and volume
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(tenkan_sen_6h[i]) or np.isnan(kijun_sen_6h[i]) or 
            np.isnan(senkou_span_a_6h[i]) or np.isnan(senkou_span_b_6h[i]) or
            np.isnan(adx_6h_aligned[i]) or np.isnan(vol_avg_6h[i])):
            if position != 0:
                # Simple stoploss: 2.5x ATR estimate
                atr_est = np.abs(high_6h[i] - low_6h[i])
                if position == 1 and close_6h[i] <= entry_price - 2.5 * atr_est:
                    signals[i] = 0.0
                    position = 0
                elif position == -1 and close_6h[i] >= entry_price + 2.5 * atr_est:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25 if position == 1 else -0.25
            continue
        
        # Determine cloud boundaries (Senkou Span A/B)
        upper_cloud = np.maximum(senkou_span_a_6h[i], senkou_span_b_6h[i])
        lower_cloud = np.minimum(senkou_span_a_6h[i], senkou_span_b_6h[i])
        
        # Trend regime: ADX > 25 = trending
        is_trend = adx_6h_aligned[i] > 25
        
        # Volume confirmation: current volume > 1.5x 6h average
        vol_confirm = volume_6h[i] > 1.5 * vol_avg_6h[i]
        
        if position == 0:
            # Look for entries
            if is_trend and vol_confirm:
                # Bullish: Tenkan > Kijun AND price above cloud
                if tenkan_sen_6h[i] > kijun_sen_6h[i] and close_6h[i] > upper_cloud:
                    signals[i] = 0.25  # long
                    position = 1
                    entry_price = close_6h[i]
                # Bearish: Tenkan < Kijun AND price below cloud
                elif tenkan_sen_6h[i] < kijun_sen_6h[i] and close_6h[i] < lower_cloud:
                    signals[i] = -0.25  # short
                    position = -1
                    entry_price = close_6h[i]
        else:
            # Manage existing position
            if position == 1:
                # Long position: exit if Tenkan < Kijun OR price below cloud
                if tenkan_sen_6h[i] < kijun_sen_6h[i] or close_6h[i] < lower_cloud:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Short position: exit if Tenkan > Kijun OR price above cloud
                if tenkan_sen_6h[i] > kijun_sen_6h[i] or close_6h[i] > upper_cloud:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals