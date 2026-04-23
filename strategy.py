#!/usr/bin/env python3
"""
Hypothesis: 6h Ichimoku Cloud breakout with 1d TK Cross filter and volume spike confirmation.
Long when price breaks above 1d Ichimoku Cloud (Senkou Span A/B) AND TK Cross bullish (Tenkan > Kijun) AND volume > 2.0x 20-period MA.
Short when price breaks below 1d Ichimoku Cloud AND TK Cross bearish (Tenkan < Kijun) AND volume > 2.0x 20-period MA.
Exit when price returns to opposite cloud boundary or TK Cross reverses.
Ichimoku Cloud provides dynamic support/resistance; TK Cross confirms momentum alignment.
Designed for ~15-25 trades/year with structure edge in trending markets.
Works in both bull (cloud acts as support in uptrend) and bear (cloud acts as resistance in downtrend).
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
    
    # Calculate 1d Ichimoku components
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # need at least 52 periods for Ichimoku
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period_tenkan = 9
    high_tenkan = pd.Series(high_1d).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    low_tenkan = pd.Series(low_1d).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (high_tenkan + low_tenkan) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period_kijun = 26
    high_kijun = pd.Series(high_1d).rolling(window=period_kijun, min_periods=period_kijun).max().values
    low_kijun = pd.Series(low_1d).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (high_kijun + low_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2 shifted 26 periods ahead
    period_senkou_b = 52
    high_senkou_b = pd.Series(high_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    low_senkou_b = pd.Series(low_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = ((high_senkou_b + low_senkou_b) / 2)
    
    # Align Ichimoku components to 6h timeframe (with proper delay for forward-shifted spans)
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a, additional_delay_bars=26)  # Leading span A is shifted 26 periods ahead
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b, additional_delay_bars=26)  # Leading span B is shifted 26 periods ahead
    
    # Calculate TK Cross: Tenkan > Kijun = bullish, Tenkan < Kijun = bearish
    tk_bullish = tenkan_aligned > kijun_aligned
    tk_bearish = tenkan_aligned < kijun_aligned
    
    # Cloud boundaries: Senkou Span A and B form the cloud
    # Upper cloud boundary = max(Senkou A, Senkou B)
    # Lower cloud boundary = min(Senkou A, Senkou B)
    upper_cloud = np.maximum(senkou_a_aligned, senkou_b_aligned)
    lower_cloud = np.minimum(senkou_a_aligned, senkou_b_aligned)
    
    # Calculate volume MA (20-period) for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(52, 20)  # need Ichimoku (52) and volume MA (20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(upper_cloud[i]) or np.isnan(lower_cloud[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Ichimoku breakout conditions
        breakout_above_cloud = close[i] > upper_cloud[i]  # Break above cloud
        breakout_below_cloud = close[i] < lower_cloud[i]  # Break below cloud
        return_to_upper_cloud = close[i] < upper_cloud[i]  # Return below upper cloud (exit long)
        return_to_lower_cloud = close[i] > lower_cloud[i]  # Return above lower cloud (exit short)
        tk_reverse = (position == 1 and tk_bearish[i]) or (position == -1 and tk_bullish[i])  # TK Cross reverse
        
        if position == 0:
            # Long: Break above cloud AND bullish TK Cross AND volume confirmation
            if breakout_above_cloud and tk_bullish[i] and volume[i] > 2.0 * vol_ma_20[i]:
                signals[i] = 0.25
                position = 1
            # Short: Break below cloud AND bearish TK Cross AND volume confirmation
            elif breakout_below_cloud and tk_bearish[i] and volume[i] > 2.0 * vol_ma_20[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: return to opposite cloud boundary or TK Cross reverse
            exit_signal = False
            if position == 1:
                exit_signal = return_to_upper_cloud or tk_reverse
            elif position == -1:
                exit_signal = return_to_lower_cloud or tk_reverse
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_Ichimoku_KumoBreak_1dTKCross_VolumeSpike"
timeframe = "6h"
leverage = 1.0