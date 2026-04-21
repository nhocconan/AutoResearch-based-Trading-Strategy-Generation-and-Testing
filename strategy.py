#!/usr/bin/env python3
"""
6h_Ichimoku_TK_Cross_CloudFilter_1dTrend
Hypothesis: 6h timeframe with Ichimoku TK cross (Tenkan/Kijun) filtered by 1d cloud (Senkou Span A/B) for trend direction.
Only take longs when price above cloud, shorts when price below cloud. Uses volume confirmation (>1.5x 20-period MA) to avoid whipsaws.
Designed for low trade frequency (~15-30/year) to minimize fee drag while capturing strong trending moves in both bull/bear markets.
Discrete sizing (0.25) with ATR(14) stoploss (2.5x). Ichimoku provides dynamic support/resistance and trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for Ichimoku cloud and trend)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d < 60):
        return np.zeros(n)
    
    # === 1d Ichimoku calculations ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Align Ichimoku components to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # === 6h ATR (14-period) for stoploss ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # === Volume confirmation (1.5x 20-period MA) ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) 
            or np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) 
            or np.isnan(atr[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        volume_now = volume[i]
        tenkan_val = tenkan_aligned[i]
        kijun_val = kijun_aligned[i]
        senkou_a_val = senkou_a_aligned[i]
        senkou_b_val = senkou_b_aligned[i]
        vol_avg = vol_ma[i]
        
        # Cloud boundaries: Senkou Span A and B form the cloud
        cloud_top = max(senkou_a_val, senkou_b_val)
        cloud_bottom = min(senkou_a_val, senkou_b_val)
        
        # Volume confirmation: current volume > 1.5x average
        volume_confirmed = volume_now > 1.5 * vol_avg
        
        if position == 0:
            # TK cross: Tenkan crosses above Kijun (bullish) or below (bearish)
            tk_cross_bullish = tenkan_val > kijun_val
            tk_cross_bearish = tenkan_val < kijun_val
            
            # Only take longs when price above cloud, shorts when price below cloud
            price_above_cloud = price > cloud_top
            price_below_cloud = price < cloud_bottom
            
            # Enter long: bullish TK cross + price above cloud + volume confirmation
            if tk_cross_bullish and price_above_cloud and volume_confirmed:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Enter short: bearish TK cross + price below cloud + volume confirmation
            elif tk_cross_bearish and price_below_cloud and volume_confirmed:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Check stoploss (2.5x ATR)
            if price < entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit if price falls below cloud (trend change)
            elif price < cloud_bottom:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Check stoploss (2.5x ATR)
            if price > entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit if price rises above cloud (trend change)
            elif price > cloud_top:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_TK_Cross_CloudFilter_1dTrend"
timeframe = "6h"
leverage = 1.0