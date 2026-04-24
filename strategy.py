#!/usr/bin/env python3
"""
Hypothesis: 6h Ichimoku Cloud with 1d TK Cross filter and volume spike confirmation.
- Primary timeframe: 6h for entries/exits.
- HTF: 1d TK Cross (Tenkan/Kijun cross) for trend direction (bullish if Tenkan > Kijun, bearish if Tenkan < Kijun).
- Volume: Current 6h volume > 2.0 * 20-period volume MA to avoid false breakouts.
- Entry: Long when price is above Ichimoku cloud AND 1d TK Cross bullish AND volume spike.
         Short when price is below Ichimoku cloud AND 1d TK Cross bearish AND volume spike.
- Exit: Opposite cloud condition or loss of volume confirmation.
- Signal size: 0.25 discrete to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
Ichimoku works in both bull and bear markets by combining trend, momentum, and support/resistance.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Ichimoku components on 6h
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2 shifted 26 periods ahead
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    # The actual cloud is Senkou A and B plotted 26 periods ahead
    # For current price, we compare with Senkou A and B from 26 periods ago
    senkou_a_lagged = np.roll(senkou_a, 26)
    senkou_b_lagged = np.roll(senkou_b, 26)
    # Set first 26 values to NaN since they don't have lagged cloud data
    senkou_a_lagged[:26] = np.nan
    senkou_b_lagged[:26] = np.nan
    
    # Get 1d data for TK Cross
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 26:
        return np.zeros(n)
    
    # Calculate 1d Tenkan and Kijun for TK Cross
    df_1d_high = df_1d['high'].values
    df_1d_low = df_1d['low'].values
    df_1d_close = df_1d['close'].values
    df_1d_volume = df_1d['volume'].values
    
    # 1d Tenkan-sen (9-period)
    period9_high_1d = pd.Series(df_1d_high).rolling(window=9, min_periods=9).max().values
    period9_low_1d = pd.Series(df_1d_low).rolling(window=9, min_periods=9).min().values
    tenkan_1d = (period9_high_1d + period9_low_1d) / 2
    
    # 1d Kijun-sen (26-period)
    period26_high_1d = pd.Series(df_1d_high).rolling(window=26, min_periods=26).max().values
    period26_low_1d = pd.Series(df_1d_low).rolling(window=26, min_periods=26).min().values
    kijun_1d = (period26_high_1d + period26_low_1d) / 2
    
    # TK Cross: 1 if bullish (Tenkan > Kijun), -1 if bearish (Tenkan < Kijun), 0 otherwise
    tk_cross = np.where(tenkan_1d > kijun_1d, 1, np.where(tenkan_1d < kijun_1d, -1, 0))
    
    # Calculate 20-period volume MA on 1d
    vol_ma_1d = pd.Series(df_1d_volume).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 6h
    tk_cross_aligned = align_htf_to_ltf(prices, df_1d, tk_cross)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Volume confirmation: current 6h volume > 2.0 * 20-period 1d volume MA (aligned)
    volume_spike = volume > (2.0 * vol_ma_1d_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(52, 26, 20)  # Need enough bars for Ichimoku and 1d indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(senkou_a_lagged[i]) or np.isnan(senkou_b_lagged[i]) or
            np.isnan(tk_cross_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        upper_cloud = max(senkou_a_lagged[i], senkou_b_lagged[i])
        lower_cloud = min(senkou_a_lagged[i], senkou_b_lagged[i])
        tk_val = tk_cross_aligned[i]
        
        if position == 0:
            # Check for entry signals with volume spike
            if volume_spike[i]:
                # Bullish: price above cloud AND 1d TK Cross bullish
                if curr_low > upper_cloud and tk_val == 1:
                    signals[i] = 0.25
                    position = 1
                # Bearish: price below cloud AND 1d TK Cross bearish
                elif curr_high < lower_cloud and tk_val == -1:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price breaks below cloud OR loss of volume confirmation
            if curr_high < lower_cloud or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above cloud OR loss of volume confirmation
            if curr_low > upper_cloud or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_IchimokuCloud_1dTKCross_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0