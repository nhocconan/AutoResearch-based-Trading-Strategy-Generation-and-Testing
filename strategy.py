#!/usr/bin/env python3
"""
Hypothesis: 6h Ichimoku Cloud strategy with 1d trend filter and volume confirmation.
Long when price is above Ichimoku cloud (Senkou Span A/B) AND Tenkan > Kijun (bullish TK cross) 
AND price > 1d EMA50 (uptrend) AND volume > 1.5x average.
Short when price is below Ichimoku cloud AND Tenkan < Kijun (bearish TK cross) 
AND price < 1d EMA50 (downtrend) AND volume > 1.5x average.
Exit when price re-enters the cloud or TK cross reverses.
Uses 6h timeframe to target ~15-30 trades/year, minimizing fee drag while capturing medium-term trends.
Ichimoku provides dynamic support/resistance via cloud, works in both bull and bear markets by requiring 
alignment with 1d EMA50 trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for EMA50 trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate EMA50 for 1d trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Ichimoku components on 6h timeframe
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = (period52_high + period52_low) / 2
    
    # Volume average (20-period) on 6h timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(52, n):  # Start after warmup period (need 52 for Senkou B)
        # Skip if data not ready
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or np.isnan(senkou_a[i]) or 
            np.isnan(senkou_b[i]) or np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine cloud boundaries (Senkou Span A/B)
        upper_cloud = max(senkou_a[i], senkou_b[i])
        lower_cloud = min(senkou_a[i], senkou_b[i])
        
        ema50_val = ema50_1d_aligned[i]
        vol_ma_val = vol_ma[i]
        vol_current = volume[i]
        price = close[i]
        tenkan_val = tenkan[i]
        kijun_val = kijun[i]
        
        if position == 0:
            # Long: price above cloud AND bullish TK cross AND price > 1d EMA50 AND volume spike
            if (price > upper_cloud and tenkan_val > kijun_val and 
                price > ema50_val and vol_current > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: price below cloud AND bearish TK cross AND price < 1d EMA50 AND volume spike
            elif (price < lower_cloud and tenkan_val < kijun_val and 
                  price < ema50_val and vol_current > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price re-enters cloud (below upper cloud) OR bearish TK cross
                if price < upper_cloud or tenkan_val < kijun_val:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price re-enters cloud (above lower cloud) OR bullish TK cross
                if price > lower_cloud or tenkan_val > kijun_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_Ichimoku_Cloud_TK_Cross_1dEMA50_Volume"
timeframe = "6h"
leverage = 1.0