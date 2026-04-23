#!/usr/bin/env python3
"""
Hypothesis: 6h Ichimoku Cloud breakout with 1d trend filter and volume confirmation.
Long when price breaks above 6h Ichimoku Cloud (Senkou Span A/B) AND Tenkan > Kijun (bullish momentum) 
AND price > 1d EMA50 (uptrend) AND volume > 2.0x average.
Short when price breaks below 6h Ichimoku Cloud AND Tenkan < Kijun (bearish momentum)
AND price < 1d EMA50 (downtrend) AND volume > 2.0x average.
Exit when price re-enters the Ichimoku Cloud OR trend reverses (price crosses 1d EMA50).
Uses 6h timeframe with Ichimoku Cloud as dynamic support/resistance to limit false breakouts.
1d EMA50 provides smooth trend filter. Volume spike ensures high-conviction breakouts.
Target: 80-140 trades over 4 years (20-35/year) to stay within proven working range for 6h.
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
    
    # Calculate 6h Ichimoku components - ONCE before loop
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 52:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period_tenkan = 9
    max_high_9 = pd.Series(high_6h).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    min_low_9 = pd.Series(low_6h).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (max_high_9 + min_low_9) / 2.0
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period_kijun = 26
    max_high_26 = pd.Series(high_6h).rolling(window=period_kijun, min_periods=period_kijun).max().values
    min_low_26 = pd.Series(low_6h).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (max_high_26 + min_low_26) / 2.0
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2.0
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period_senkou_b = 52
    max_high_52 = pd.Series(high_6h).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    min_low_52 = pd.Series(low_6h).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = (max_high_52 + min_low_52) / 2.0
    
    # Load 1d data for EMA50 trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF indicators to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_6h, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_6h, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_6h, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_6h, senkou_b)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume average (20-period) on 6h timeframe
    vol_6h = df_6h['volume'].values
    vol_ma_6h = pd.Series(vol_6h).rolling(window=20, min_periods=20).mean().values
    vol_ma_6h_aligned = align_htf_to_ltf(prices, df_6h, vol_ma_6h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if data not ready
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or np.isnan(senkou_a_aligned[i]) or 
            np.isnan(senkou_b_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma_6h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        tenkan_val = tenkan_aligned[i]
        kijun_val = kijun_aligned[i]
        senkou_a_val = senkou_a_aligned[i]
        senkou_b_val = senkou_b_aligned[i]
        ema50_val = ema50_1d_aligned[i]
        vol_ma_val = vol_ma_6h_aligned[i]
        
        # Determine Ichimoku Cloud boundaries (upper and lower bands)
        upper_cloud = max(senkou_a_val, senkou_b_val)
        lower_cloud = min(senkou_a_val, senkou_b_val)
        
        # Get current 6h-aligned price and volume
        price = close[i]
        vol_current = volume[i]
        vol_ma_primary = pd.Series(volume).rolling(window=20, min_periods=20).mean().values[i]
        
        if position == 0:
            # Long: price breaks above Ichimoku Cloud AND Tenkan > Kijun (bullish) AND price > 1d EMA50 AND volume spike
            if (price > upper_cloud and tenkan_val > kijun_val and price > ema50_val and vol_current > 2.0 * vol_ma_primary):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Ichimoku Cloud AND Tenkan < Kijun (bearish) AND price < 1d EMA50 AND volume spike
            elif (price < lower_cloud and tenkan_val < kijun_val and price < ema50_val and vol_current > 2.0 * vol_ma_primary):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price re-enters Ichimoku Cloud (below upper cloud) OR price breaks below 1d EMA50 (trend reversal)
                if price < upper_cloud or price < ema50_val:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price re-enters Ichimoku Cloud (above lower cloud) OR price breaks above 1d EMA50 (trend reversal)
                if price > lower_cloud or price > ema50_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_Ichimoku_Cloud_Breakout_1dEMA50_Volume"
timeframe = "6h"
leverage = 1.0