#!/usr/bin/env python3
"""
Hypothesis: 6h Ichimoku Cloud with 1d filter and volume confirmation
- Uses Ichimoku Cloud (Tenkan-sen, Kijun-sen, Senkou Span A/B) on 6h as primary trend and momentum indicator
- 1d EMA50 as HTF trend filter to ensure alignment with daily trend
- Volume spike (2.0x 20-period MA) confirms breakout strength and reduces false signals
- Long when price > Cloud + Tenkan > Kijun + volume spike + price > 1d EMA50 (bullish alignment)
- Short when price < Cloud + Tenkan < Kijun + volume spike + price < 1d EMA50 (bearish alignment)
- Ichimoku works in both trending and ranging markets; cloud acts as dynamic support/resistance
- Discrete position sizing (0.25) minimizes fee churn
- Target: 12-37 trades/year per symbol (~50-150 total over 4 years)
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
    
    # Get 6h data for Ichimoku calculation (primary timeframe)
    df_6h = get_htf_data(prices, '6h')
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Get 1d data for EMA50 trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate Ichimoku components on 6h
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high_6h).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_6h).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high_6h).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_6h).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2, plotted 26 periods ahead
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2, plotted 26 periods ahead
    period52_high = pd.Series(high_6h).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_6h).rolling(window=52, min_periods=52).min().values
    senkou_b = (period52_high + period52_low) / 2
    
    # Calculate EMA50 on 1d for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume average (20-period) on 6h
    volume_6h = df_6h['volume'].values
    volume_ma_20 = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to 6h timeframe (primary)
    tenkan_aligned = align_htf_to_ltf(prices, df_6h, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_6h, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_6h, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_6h, senkou_b)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    volume_ma_aligned = align_htf_to_ltf(prices, df_6h, volume_ma_20)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # warmup
    
    for i in range(start_idx, n):
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(volume_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        tenkan_val = tenkan_aligned[i]
        kijun_val = kijun_aligned[i]
        senkou_a_val = senkou_a_aligned[i]
        senkou_b_val = senkou_b_aligned[i]
        ema_trend = ema50_1d_aligned[i]
        vol_ma = volume_ma_aligned[i]
        vol = volume[i]
        price = close[i]
        
        # Determine cloud boundaries (Senkou Span A/B)
        upper_cloud = max(senkou_a_val, senkou_b_val)
        lower_cloud = min(senkou_a_val, senkou_b_val)
        
        if position == 0:
            # Long: price above cloud + Tenkan > Kijun (bullish momentum) + volume spike + price > 1d EMA50
            if (price > upper_cloud and tenkan_val > kijun_val and 
                vol > 2.0 * vol_ma and price > ema_trend):
                signals[i] = 0.25
                position = 1
            # Short: price below cloud + Tenkan < Kijun (bearish momentum) + volume spike + price < 1d EMA50
            elif (price < lower_cloud and tenkan_val < kijun_val and 
                  vol > 2.0 * vol_ma and price < ema_trend):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls below cloud OR Tenkan < Kijun (momentum shift)
            if price < lower_cloud or tenkan_val < kijun_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises above cloud OR Tenkan > Kijun (momentum shift)
            if price > upper_cloud or tenkan_val > kijun_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_1dEMA50_VolumeSpike"
timeframe = "6h"
leverage = 1.0