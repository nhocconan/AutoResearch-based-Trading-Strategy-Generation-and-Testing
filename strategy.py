#!/usr/bin/env python3
"""
6h_Ichimoku_Kumo_Twist_V1
Hypothesis: 6h Ichimoku cloud twist (Tenkan/Kijun cross) with 1d cloud color filter (green/red) 
captures trend reversals with alignment to higher timeframe trend. 
Volume confirmation (>1.3x 20-period average) filters false signals. 
ATR(14) trailing stop via signal=0 when price moves against position by 2.0*ATR. 
Designed for low trade frequency (target: 12-25 trades/year) to minimize fee drag 
and work in both bull/bear markets via cloud twist + higher timeframe cloud color alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for cloud color filter)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    # === 1d Ichimoku components for cloud color ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan_sen_1d = (pd.Series(high_1d).rolling(window=9, min_periods=9).max() + 
                     pd.Series(low_1d).rolling(window=9, min_periods=9).min()) / 2
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_sen_1d = (pd.Series(high_1d).rolling(window=26, min_periods=26).max() + 
                    pd.Series(low_1d).rolling(window=26, min_periods=26).min()) / 2
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_span_a_1d = ((tenkan_sen_1d + kijun_sen_1d) / 2).shift(26)
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    senkou_span_b_1d = ((pd.Series(high_1d).rolling(window=52, min_periods=52).max() + 
                         pd.Series(low_1d).rolling(window=52, min_periods=52).min()) / 2).shift(26)
    # Cloud color: green when Span A > Span B, red when Span A < Span B
    cloud_green_1d = senkou_span_a_1d > senkou_span_b_1d
    cloud_red_1d = senkou_span_a_1d < senkou_span_b_1d
    
    # Align 1d Ichimoku components to 6h
    tenkan_sen_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen_1d.values)
    kijun_sen_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen_1d.values)
    senkou_span_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a_1d.values)
    senkou_span_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b_1d.values)
    cloud_green_1d_aligned = align_htf_to_ltf(prices, df_1d, cloud_green_1d.astype(float).values)
    cloud_red_1d_aligned = align_htf_to_ltf(prices, df_1d, cloud_red_1d.astype(float).values)
    
    # === 6h Ichimoku components (primary timeframe) ===
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    volume_6h = prices['volume'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan_sen_6h = (pd.Series(high_6h).rolling(window=9, min_periods=9).max() + 
                     pd.Series(low_6h).rolling(window=9, min_periods=9).min()) / 2
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_sen_6h = (pd.Series(high_6h).rolling(window=26, min_periods=26).max() + 
                    pd.Series(low_6h).rolling(window=26, min_periods=26).min()) / 2
    # Chikou Span (Lagging Shift): close shifted -22 periods (not used for signals)
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_span_a_6h = ((tenkan_sen_6h + kijun_sen_6h) / 2).shift(26)
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    senkou_span_b_6h = ((pd.Series(high_6h).rolling(window=52, min_periods=52).max() + 
                         pd.Series(low_6h).rolling(window=52, min_periods=52).min()) / 2).shift(26)
    
    # Volume confirmation: current volume > 1.3x 20-period average
    vol_ma = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.3 * vol_ma
    
    # ATR (14-period) for stoploss
    tr1 = pd.Series(high_6h - low_6h)
    tr2 = pd.Series(np.abs(high_6h - np.roll(close_6h, 1)))
    tr3 = pd.Series(np.abs(low_6h - np.roll(close_6h, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(60, n):
        # Skip if indicators not ready
        if (np.isnan(tenkan_sen_6h[i]) or np.isnan(kijun_sen_6h[i]) 
            or np.isnan(senkou_span_a_6h[i]) or np.isnan(senkou_span_b_6h[i])
            or np.isnan(volume_threshold[i]) or np.isnan(atr[i])
            or np.isnan(tenkan_sen_1d_aligned[i]) or np.isnan(kijun_sen_1d_aligned[i])
            or np.isnan(cloud_green_1d_aligned[i]) or np.isnan(cloud_red_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_6h[i]
        
        if position == 0:
            # Bullish TK cross + price above cloud + 1d cloud green + volume confirmation
            tk_cross_bullish = tenkan_sen_6h[i] > kijun_sen_6h[i] and tenkan_sen_6h[i-1] <= kijun_sen_6h[i-1]
            price_above_cloud = price > max(senkou_span_a_6h[i], senkou_span_b_6h[i])
            if tk_cross_bullish and price_above_cloud and cloud_green_1d_aligned[i] > 0.5 and volume_6h[i] > volume_threshold[i]:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Bearish TK cross + price below cloud + 1d cloud red + volume confirmation
            elif tk_cross_bullish is False and tenkan_sen_6h[i] < kijun_sen_6h[i] and tenkan_sen_6h[i-1] >= kijun_sen_6h[i-1]:
                tk_cross_bearish = True
                price_below_cloud = price < min(senkou_span_a_6h[i], senkou_span_b_6h[i])
                if tk_cross_bearish and price_below_cloud and cloud_red_1d_aligned[i] > 0.5 and volume_6h[i] > volume_threshold[i]:
                    signals[i] = -0.25
                    position = -1
                    entry_price = price
        
        elif position == 1:
            # Check stoploss
            if price < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trailing exit: TK cross turns bearish or price falls below cloud
            elif (tenkan_sen_6h[i] < kijun_sen_6h[i] and tenkan_sen_6h[i-1] >= kijun_sen_6h[i-1]) or \
                 price < min(senkou_span_a_6h[i], senkou_span_b_6h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Check stoploss
            if price > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trailing exit: TK cross turns bullish or price rises above cloud
            elif (tenkan_sen_6h[i] > kijun_sen_6h[i] and tenkan_sen_6h[i-1] <= kijun_sen_6h[i-1]) or \
                 price > max(senkou_span_a_6h[i], senkou_span_b_6h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Kumo_Twist_V1"
timeframe = "6h"
leverage = 1.0