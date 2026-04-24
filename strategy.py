#!/usr/bin/env python3
"""
Hypothesis: 6h Ichimoku Cloud with 1d TK Cross and Volume Spike Filter
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for Ichimoku components (Tenkan, Kijun, Senkou Span A/B) and volume confirmation.
- Ichimoku Cloud: Provides dynamic support/resistance and trend direction.
- Entry: Long when price > Senkou Span A AND Tenkan > Kijun (TK cross bullish) AND volume > 2.0 * 20-period average volume.
         Short when price < Senkou Span B AND Tenkan < Kijun (TK cross bearish) AND volume > 2.0 * 20-period average volume.
- Exit: Opposite TK cross OR price crosses opposite Senkou Span (A for long, B for short).
- Signal size: 0.25 discrete to minimize fee drag.
- Ichimoku works in both trending and ranging markets by identifying trend, momentum, and support/resistance.
- 1d HTF ensures alignment with daily structure, reducing noise on 6h chart.
- Volume confirmation filters low-participation breakouts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def ichimoku_components(high, low, close):
    """Calculate Ichimoku Cloud components: Tenkan, Kijun, Senkou A, Senkou B."""
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max()
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min()
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max()
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min()
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max()
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min()
    senkou_b = ((period52_high + period52_low) / 2)
    
    return tenkan.values, kijun.values, senkou_a.values, senkou_b.values

def generate_signals(prices):
    n = len(prices)
    if n < 60:  # Need sufficient data for Ichimoku (52-period lookback)
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d Ichimoku components
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 55:  # Need sufficient data for Ichimoku (52-period lookback)
        return np.zeros(n)
    
    tenkan_1d, kijun_1d, senkou_a_1d, senkou_b_1d = ichimoku_components(
        df_1d['high'].values, 
        df_1d['low'].values, 
        df_1d['close'].values
    )
    
    # Align Ichimoku components to 6h timeframe (no additional delay needed for Ichimoku)
    tenkan_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d)
    kijun_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d)
    senkou_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_1d)
    senkou_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_1d)
    
    # Calculate 1d volume average for confirmation
    if len(df_1d) < 21:
        return np.zeros(n)
    
    vol_ma_20 = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1d = df_1d['volume'].values / (vol_ma_20 + 1e-10)  # Avoid division by zero
    vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 60  # Need sufficient data for Ichimoku (52-period lookback)
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(tenkan_1d_aligned[i]) or np.isnan(kijun_1d_aligned[i]) or
            np.isnan(senkou_a_1d_aligned[i]) or np.isnan(senkou_b_1d_aligned[i]) or
            np.isnan(vol_ratio_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        curr_tenkan = tenkan_1d_aligned[i]
        curr_kijun = kijun_1d_aligned[i]
        curr_senkou_a = senkou_a_1d_aligned[i]
        curr_senkou_b = senkou_b_1d_aligned[i]
        
        # Exit conditions: Opposite TK cross OR price crosses opposite Senkou Span
        if position != 0:
            # Exit long: Tenkan < Kijun (bearish TK cross) OR price falls below Senkou Span A
            if position == 1:
                if curr_tenkan < curr_kijun or curr_close < curr_senkou_a:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: Tenkan > Kijun (bullish TK cross) OR price rises above Senkou Span B
            elif position == -1:
                if curr_tenkan > curr_kijun or curr_close > curr_senkou_b:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: TK cross with cloud filter and volume confirmation
        if position == 0:
            # Bullish TK cross: Tenkan crosses above Kijun
            tk_bullish_cross = curr_tenkan > curr_kijun and (i == start_idx or tenkan_1d_aligned[i-1] <= kijun_1d_aligned[i-1])
            # Bearish TK cross: Tenkan crosses below Kijun
            tk_bearish_cross = curr_tenkan < curr_kijun and (i == start_idx or tenkan_1d_aligned[i-1] >= kijun_1d_aligned[i-1])
            
            # Long: Bullish TK cross AND price > Senkou Span A (above cloud) AND volume confirmation
            if tk_bullish_cross and curr_close > curr_senkou_a and curr_volume > 2.0 * vol_ma_20[min(i, len(vol_ma_20)-1)] if len(vol_ma_20) > 0 else False:
                signals[i] = 0.25
                position = 1
            # Short: Bearish TK cross AND price < Senkou Span B (below cloud) AND volume confirmation
            elif tk_bearish_cross and curr_close < curr_senkou_b and curr_volume > 2.0 * vol_ma_20[min(i, len(vol_ma_20)-1)] if len(vol_ma_20) > 0 else False:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_1dTK_Cross_CloudFilter_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0