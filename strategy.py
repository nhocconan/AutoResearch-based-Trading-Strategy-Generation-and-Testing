#!/usr/bin/env python3
name = "1d_Ichimoku_Tenkan_Kijun_Cross_WeeklyTrend"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate Ichimoku components (Tenkan-sen and Kijun-sen)
    # Tenkan-sen: (9-period high + 9-period low) / 2
    high_9 = pd.Series(high).rolling(window=9, min_periods=9).max().values
    low_9 = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (high_9 + low_9) / 2
    
    # Kijun-sen: (26-period high + 26-period low) / 2
    high_26 = pd.Series(high).rolling(window=26, min_periods=26).max().values
    low_26 = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (high_26 + low_26) / 2
    
    # Weekly EMA for trend filter
    ema_20_1w = pd.Series(df_1w['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Volume spike detection
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(60, 26)  # Need enough data for Ichimoku
    
    for i in range(start_idx, n):
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(ema_20_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Ichimoku cross signals
        tenkan_above_kijun = tenkan[i] > kijun[i]
        tenkan_below_kijun = tenkan[i] < kijun[i]
        
        vol_condition = volume[i] > vol_ma_20[i] * 1.5
        
        if position == 0:
            # Long: Tenkan crosses above Kijun in weekly uptrend with volume
            if tenkan_above_kijun and tenkan[i-1] <= kijun[i-1] and vol_condition and ema_20_1w_aligned[i] > ema_20_1w_aligned[i-1]:
                signals[i] = 0.25
                position = 1
            # Short: Tenkan crosses below Kijun in weekly downtrend with volume
            elif tenkan_below_kijun and tenkan[i-1] >= kijun[i-1] and vol_condition and ema_20_1w_aligned[i] < ema_20_1w_aligned[i-1]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Tenkan crosses back below Kijun
            if tenkan_below_kijun and tenkan[i-1] >= kijun[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Tenkan crosses back above Kijun
            if tenkan_above_kijun and tenkan[i-1] <= kijun[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 1d Ichimoku Tenkan/Kijun cross with weekly trend filter and volume confirmation
# - Tenkan-sen (9-period) crossing above Kijun-sen (26-period) = bullish signal
# - Tenkan-sen crossing below Kijun-sen = bearish signal
# - Weekly EMA20 trend filter ensures alignment with higher timeframe trend
# - Volume confirmation (1.5x average) reduces false signals
# - Works in both bull and bear markets by following weekly trend
# - Ichimoku provides clear trend-following signals with built-in support/resistance
# - Position size 0.25 targets ~20-50 trades/year to avoid fee drag
# - Weekly trend filter reduces whipsaws vs same-timeframe signals
# - Ichimoku is a proven trend-following system that works across market regimes
# - Aims for 80-200 total trades over 4 years (20-50/year) to stay within limits