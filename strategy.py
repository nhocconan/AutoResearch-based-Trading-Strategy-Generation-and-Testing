#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud breakout with 1d trend filter and volume confirmation.
# Long when price breaks above the 1d Ichimoku cloud (Senkou Span A/B) AND Tenkan > Kijun AND 6h volume > 1.5x 20-period volume MA.
# Short when price breaks below the 1d Ichimoku cloud AND Tenkan < Kijun AND 6h volume > 1.5x 20-period volume MA.
# Uses 1d Ichimoku for higher-timeframe structure, which adapts to both bull and bear markets via cloud color and price position.
# Exit when price re-enters the cloud or Tenkan/Kijun cross reverses.
# Position size 0.25 to manage drawdown in volatile markets like 2022.
# Designed for 6h timeframe to achieve 50-150 total trades over 4 years (12-37/year) with strict entry conditions.

name = "6h_Ichimoku_Cloud_Breakout_1dTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (avoid low-liquidity periods)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for Ichimoku calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # Need at least 52 periods for Ichimoku (26*2)
        return np.zeros(n)
    
    # Calculate 1d Ichimoku components
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period_tenkan = 9
    max_high_tenkan = pd.Series(high_1d).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    min_low_tenkan = pd.Series(low_1d).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (max_high_tenkan + min_low_tenkan) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period_kijun = 26
    max_high_kijun = pd.Series(high_1d).rolling(window=period_kijun, min_periods=period_kijun).max().values
    min_low_kijun = pd.Series(low_1d).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (max_high_kijun + min_low_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period_senkou_b = 52
    max_high_senkou_b = pd.Series(high_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    min_low_senkou_b = pd.Series(low_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = ((max_high_senkou_b + min_low_senkou_b) / 2)
    
    # Align Ichimoku components to 6h timeframe (with proper shift for leading spans)
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # Calculate 6h volume 20-period MA for spike detection
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(52, n):  # Start after Ichimoku warmup
        # Skip if any value is NaN or outside session
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or 
            np.isnan(volume_ma_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        # Cloud boundaries: Senkou Span A and B form the cloud
        cloud_top = max(senkou_a_aligned[i], senkou_b_aligned[i])
        cloud_bottom = min(senkou_a_aligned[i], senkou_b_aligned[i])
        
        # Volume spike condition: current 6h volume > 1.5x 20-period volume MA
        volume_spike = volume[i] > (volume_ma_20[i] * 1.5)
        
        # Ichimoku trend conditions: Tenkan > Kijun for uptrend, Tenkan < Kijun for downtrend
        tenkan_above_kijun = tenkan_aligned[i] > kijun_aligned[i]
        tenkan_below_kijun = tenkan_aligned[i] < kijun_aligned[i]
        
        if position == 0:
            # Long: Price breaks above cloud TOP AND Tenkan > Kijun AND volume spike AND session
            if close_val > cloud_top and tenkan_above_kijun and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below cloud BOTTOM AND Tenkan < Kijun AND volume spike AND session
            elif close_val < cloud_bottom and tenkan_below_kijun and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price re-enters cloud (below cloud top) OR Tenkan/Kijun cross reverses
            if close_val < cloud_top or not tenkan_above_kijun:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price re-enters cloud (above cloud bottom) OR Tenkan/Kijun cross reverses
            if close_val > cloud_bottom or not tenkan_below_kijun:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals