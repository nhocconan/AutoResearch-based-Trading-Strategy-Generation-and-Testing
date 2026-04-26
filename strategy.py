#!/usr/bin/env python3
"""
6h_Ichimoku_Kumo_Twist_ADX_Filter_v1
Hypothesis: 6h Ichimoku cloud twist (Senkou Span A/B cross) with ADX trend strength filter from 1d.
Long when price > cloud, Senkou Span A > Senkou Span B (bullish twist), and ADX > 25.
Short when price < cloud, Senkou Span A < Senkou Span B (bearish twist), and ADX > 25.
Uses 1d Ichimoku for higher timeframe structure, reducing false signals in choppy markets.
ADX filter ensures we only trade when trend is strong enough to sustain moves.
Designed for 12-37 trades/year (50-150 over 4 years) by requiring confluence of cloud twist, price/cloud position, and trend strength.
Works in bull/bear via ADX filter: only takes trades when trend is strong, avoiding whipsaws in ranging markets.
Uses discrete position sizing (0.25) to minimize fee churn.
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
    
    # Load 1d data ONCE before loop for HTF Ichimoku and ADX
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Ichimoku components on 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period_tenkan = 9
    tenkan_sen = (pd.Series(high_1d).rolling(window=period_tenkan, min_periods=period_tenkan).max() + 
                  pd.Series(low_1d).rolling(window=period_tenkan, min_periods=period_tenkan).min()) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period_kijun = 26
    kijun_sen = (pd.Series(high_1d).rolling(window=period_kijun, min_periods=period_kijun).max() + 
                 pd.Series(low_1d).rolling(window=period_kijun, min_periods=period_kijun).min()) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period_senkou_b = 52
    senkou_span_b = ((pd.Series(high_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).max() + 
                      pd.Series(low_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).min()) / 2).shift(26)
    
    # Align Ichimoku components to 6h timeframe
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a.values)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b.values)
    
    # Calculate ADX on 1d for trend strength
    # True Range
    tr1 = pd.Series(high_1d) - pd.Series(low_1d)
    tr2 = abs(pd.Series(high_1d) - pd.Series(close_1d).shift(1))
    tr3 = abs(pd.Series(low_1d) - pd.Series(close_1d).shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    dm_plus = pd.Series(high_1d).diff()
    dm_minus = -pd.Series(low_1d).diff()
    dm_plus = dm_plus.where((dm_plus > dm_minus) & (dm_plus > 0), 0)
    dm_minus = dm_minus.where((dm_minus > dm_plus) & (dm_minus > 0), 0)
    
    # Smoothed TR, DM+, DM- (using Wilder's smoothing, equivalent to EMA with alpha=1/period)
    period_adx = 14
    atr = tr.ewm(alpha=1/period_adx, adjust=False).mean()
    dm_plus_smooth = dm_plus.ewm(alpha=1/period_adx, adjust=False).mean()
    dm_minus_smooth = dm_minus.ewm(alpha=1/period_adx, adjust=False).mean()
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / atr
    di_minus = 100 * dm_minus_smooth / atr
    
    # DX and ADX
    dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = dx.ewm(alpha=1/period_adx, adjust=False).mean()
    
    # Align ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 52+26 for Senkou Span B, 14 for ADX)
    start_idx = 52 + 26  # 78
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i]) or 
            np.isnan(adx_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Ichimoku conditions
        bullish_twist = senkou_span_a_aligned[i] > senkou_span_b_aligned[i]
        bearish_twist = senkou_span_a_aligned[i] < senkou_span_b_aligned[i]
        price_above_cloud = close[i] > max(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        price_below_cloud = close[i] < min(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        
        # ADX trend strength filter
        strong_trend = adx_aligned[i] > 25
        
        # Entry conditions
        if strong_trend:
            # Long when bullish twist and price above cloud
            if bullish_twist and price_above_cloud:
                if position != 1:
                    signals[i] = 0.25
                    position = 1
                else:
                    signals[i] = 0.25
            # Short when bearish twist and price below cloud
            elif bearish_twist and price_below_cloud:
                if position != -1:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = -0.25
            # Exit conditions: twist reversal or price crosses cloud opposite direction
            elif position == 1 and (not bullish_twist or price_below_cloud):
                signals[i] = 0.0
                position = 0
            elif position == -1 and (not bearish_twist or price_above_cloud):
                signals[i] = 0.0
                position = 0
            # Hold current position
            else:
                if position == 0:
                    signals[i] = 0.0
                elif position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
        else:
            # No strong trend: exit any position and stay flat
            if position == 1 or position == -1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_Ichimoku_Kumo_Twist_ADX_Filter_v1"
timeframe = "6h"
leverage = 1.0