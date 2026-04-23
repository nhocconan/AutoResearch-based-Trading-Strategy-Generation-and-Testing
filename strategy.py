#!/usr/bin/env python3
"""
Hypothesis: 6h Ichimoku Cloud + TK Cross + 1d ADX Trend Filter
Ichimoku provides dynamic support/resistance via the cloud (Senkou Span A/B) and momentum via TK Cross.
The 1d ADX acts as a regime filter: only take trades when ADX > 25 (trending market) to avoid whipsaws in ranging markets.
This combines trend-following (cloud breakout) with momentum (TK cross) and avoids false signals in low-volatility regimes.
Timeframe 6h balances signal quality and trade frequency. Target: 50-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate Ichimoku components on 6h data
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period_tenkan = 9
    max_high_tenkan = pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    min_low_tenkan = pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan_sen = (max_high_tenkan + min_low_tenkan) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period_kijun = 26
    max_high_kijun = pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max().values
    min_low_kijun = pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun_sen = (max_high_kijun + min_low_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period_senkou_b = 52
    max_high_senkou_b = pd.Series(high).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    min_low_senkou_b = pd.Series(low).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_span_b = ((max_high_senkou_b + min_low_senkou_b) / 2)
    
    # Chikou Span (Lagging Span): close shifted 26 periods behind (not used for signals to avoid look-ahead)
    
    # Calculate 1d ADX for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # need enough data for ADX calculation
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range (TR)
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    
    # Calculate Directional Movement (+DM and -DM)
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    up_move[0] = 0
    down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smooth TR, +DM, -DM using Wilder's smoothing (EMA with alpha=1/period)
    period_adx = 14
    alpha = 1.0 / period_adx
    
    atr = np.zeros_like(tr)
    atr[0] = tr[0]
    for i in range(1, len(tr)):
        atr[i] = (1 - alpha) * atr[i-1] + alpha * tr[i]
    
    plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=alpha, adjust=False).mean().values / atr)
    minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=alpha, adjust=False).mean().values / atr)
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(alpha=alpha, adjust=False).mean().values
    
    # Align HTF indicators to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)  # Not actually used but kept for consistency
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Align Ichimoku components (calculated on 6h data) - no alignment needed as they're already on 6h
    # But we need to shift Senkou Spans forward by 26 periods (they are plotted ahead)
    # For signal generation, we use current Senkou Span values (which were calculated 26 periods ago)
    # So we need to shift them back by 26 to align with current price
    senkou_span_a_lagged = np.roll(senkou_span_a, 26)
    senkou_span_b_lagged = np.roll(senkou_span_b, 26)
    senkou_span_a_lagged[:26] = np.nan  # first 26 values invalid
    senkou_span_b_lagged[:26] = np.nan
    
    # TK Cross: Tenkan-sen crossing above/below Kijun-sen
    tk_cross_above = (tenkan_sen > kijun_sen) & (np.roll(tenkan_sen, 1) <= np.roll(kijun_sen, 1))
    tk_cross_below = (tenkan_sen < kijun_sen) & (np.roll(tenkan_sen, 1) >= np.roll(kijun_sen, 1))
    
    # Cloud: price above/below both Senkou Spans
    cloud_top = np.maximum(senkou_span_a_lagged, senkou_span_b_lagged)
    cloud_bottom = np.minimum(senkou_span_a_lagged, senkou_span_b_lagged)
    price_above_cloud = close > cloud_top
    price_below_cloud = close < cloud_bottom
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(52, 26, 30)  # need Senkou Span B (52), Kijun (26), ADX (30)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(tenkan_sen[i]) or np.isnan(kijun_sen[i]) or 
            np.isnan(senkou_span_a_lagged[i]) or np.isnan(senkou_span_b_lagged[i]) or
            np.isnan(adx_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # ADX trend filter: only trade when ADX > 25 (trending market)
        strong_trend = adx_aligned[i] > 25
        
        if position == 0:
            # Long: TK cross bullish + price above cloud + strong trend
            if tk_cross_above[i] and price_above_cloud[i] and strong_trend:
                signals[i] = 0.25
                position = 1
            # Short: TK cross bearish + price below cloud + strong trend
            elif tk_cross_below[i] and price_below_cloud[i] and strong_trend:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            if position == 1:
                # Exit long when TK cross bearish OR price falls below cloud
                if tk_cross_below[i] or price_below_cloud[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short when TK cross bullish OR price rises above cloud
                if tk_cross_above[i] or price_above_cloud[i]:
                    exit_signal = True
            
            # Also exit if trend weakens (ADX drops below 20)
            if adx_aligned[i] < 20:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_Ichimoku_TK_Cross_Cloud_ADX_Trend_Filter"
timeframe = "6h"
leverage = 1.0