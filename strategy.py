#!/usr/bin/env python3
"""
6h_Ichimoku_TK_Cross_CloudFilter_1dTrend_v1
Hypothesis: On 6h timeframe, use Ichimoku Tenkan/Kijun cross for entry signals, filtered by 1d cloud (Senkou Span A/B) for trend direction and 1d volume confirmation to avoid false signals. In bullish 1d regime (price > cloud), only take long TK crosses; in bearish 1d regime (price < cloud), only take short TK crosses. Uses discrete sizing (0.25) targeting 12-30 trades/year to minimize fee drag on 6h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for Ichimoku cloud and volume)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    # === 1d Ichimoku components (9, 26, 52 periods) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2.0
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2.0
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2.0
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b = (period52_high + period52_low) / 2.0
    
    # === 1d volume confirmation (volume > 1.5x 20-period average) ===
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_confirmed_1d = volume_1d > (1.5 * vol_ma_20)
    
    # Align 1d indicators to 6h timeframe
    tenkan_6h = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_6h = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_6h = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_6h = align_htf_to_ltf(prices, df_1d, senkou_b)
    volume_confirmed_6h = align_htf_to_ltf(prices, df_1d, volume_confirmed_1d.astype(float))
    
    # === 6h price data ===
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i]) or 
            np.isnan(senkou_a_6h[i]) or np.isnan(senkou_b_6h[i]) or 
            np.isnan(volume_confirmed_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        tenkan_val = tenkan_6h[i]
        kijun_val = kijun_6h[i]
        senkou_a_val = senkou_a_6h[i]
        senkou_b_val = senkou_b_6h[i]
        vol_conf = volume_confirmed_6h[i] > 0.5  # boolean
        
        # Determine cloud top and bottom
        cloud_top = max(senkou_a_val, senkou_b_val)
        cloud_bottom = min(senkou_a_val, senkou_b_val)
        
        # 1d trend regime: price above/below cloud
        is_bullish_regime = price > cloud_top
        is_bearish_regime = price < cloud_bottom
        
        # TK cross conditions (using previous bar to avoid look-ahead)
        tenkan_prev = tenkan_6h[i-1]
        kijun_prev = kijun_6h[i-1]
        
        # Bullish TK cross: Tenkan crosses above Kijun
        tk_bull_cross = (tenkan_prev <= kijun_prev) and (tenkan_val > kijun_val)
        # Bearish TK cross: Tenkan crosses below Kijun
        tk_bear_cross = (tenkan_prev >= kijun_prev) and (tenkan_val < kijun_val)
        
        if position == 0:
            # Only take longs in bullish regime, shorts in bearish regime
            if is_bullish_regime and tk_bull_cross and vol_conf:
                signals[i] = 0.25
                position = 1
            elif is_bearish_regime and tk_bear_cross and vol_conf:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: TK cross in opposite direction or price exits cloud
            if position == 1:  # long position
                # Exit on bearish TK cross or price drops below cloud bottom
                if tk_bear_cross or price < cloud_bottom:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1, short position
                # Exit on bullish TK cross or price rises above cloud top
                if tk_bull_cross or price > cloud_top:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_TK_Cross_CloudFilter_1dTrend_v1"
timeframe = "6h"
leverage = 1.0