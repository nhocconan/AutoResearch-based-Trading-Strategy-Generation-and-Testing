#!/usr/bin/env python3
"""
Hypothesis: 6h Ichimoku Cloud with 1d TK Cross and Volume Confirmation.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for Ichimoku components (Tenkan, Kijun, Senkou Span A/B) and volume average.
- Ichimoku Cloud: Trend direction (price above/below cloud), momentum (TK cross), and support/resistance.
- Entry: Long when price > cloud, Tenkan > Kijun (bullish TK cross), and volume > 1.5 * 20-period average volume.
         Short when price < cloud, Tenkan < Kijun (bearish TK cross), and volume > 1.5 * 20-period average volume.
- Exit: Opposite TK cross OR price crosses cloud in opposite direction.
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- Ichimoku provides comprehensive trend, momentum, and support/resistance in one indicator, effective in both trending and ranging markets.
- 1d Ichimoku filters for higher timeframe structure, reducing noise on 6h chart.
- Volume confirmation ensures breakouts have participation, reducing false signals.
- Estimated trades: ~100 total over 4 years (~25/year) based on TK cross frequency with filters.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:  # Need sufficient data for Ichimoku calculations
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d Ichimoku components
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # Need sufficient data for Senkou Span B (52 periods)
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period_tenkan = 9
    highest_tenkan = pd.Series(high_1d).rolling(window=period_tenkan, min_periods=period_tenkan).max()
    lowest_tenkan = pd.Series(low_1d).rolling(window=period_tenkan, min_periods=period_tenkan).min()
    tenkan = ((highest_tenkan + lowest_tenkan) / 2).values
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period_kijun = 26
    highest_kijun = pd.Series(high_1d).rolling(window=period_kijun, min_periods=period_kijun).max()
    lowest_kijun = pd.Series(low_1d).rolling(window=period_kijun, min_periods=period_kijun).min()
    kijun = ((highest_kijun + lowest_kijun) / 2).values
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period_senkou_b = 52
    highest_senkou_b = pd.Series(high_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).max()
    lowest_senkou_b = pd.Series(low_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).min()
    senkou_b = ((highest_senkou_b + lowest_senkou_b) / 2).values
    
    # Align Ichimoku components to 6h timeframe (with 1-bar completed candle delay)
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # Calculate 1d volume average for confirmation
    if len(df_1d) < 20:
        return np.zeros(n)
    
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1w = volume_1d / (vol_ma_20 + 1e-10)  # Avoid division by zero
    vol_ratio_1w_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 52  # Need sufficient data for Senkou Span B
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or
            np.isnan(vol_ratio_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        curr_tenkan = tenkan_aligned[i]
        curr_kijun = kijun_aligned[i]
        curr_senkou_a = senkou_a_aligned[i]
        curr_senkou_b = senkou_b_aligned[i]
        
        # Determine cloud top and bottom
        cloud_top = max(curr_senkou_a, curr_senkou_b)
        cloud_bottom = min(curr_senkou_a, curr_senkou_b)
        
        # Exit conditions: opposite TK cross OR price crosses cloud in opposite direction
        if position != 0:
            # Exit long: bearish TK cross (Tenkan < Kijun) OR price falls below cloud
            if position == 1:
                if curr_tenkan < curr_kijun or curr_close < cloud_bottom:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: bullish TK cross (Tenkan > Kijun) OR price rises above cloud
            elif position == -1:
                if curr_tenkan > curr_kijun or curr_close > cloud_top:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: TK cross with cloud filter and volume confirmation
        if position == 0:
            # Bullish TK cross: Tenkan crosses above Kijun
            tk_bullish_cross = curr_tenkan > curr_kijun and (i == start_idx or tenkan_aligned[i-1] <= kijun_aligned[i-1])
            # Bearish TK cross: Tenkan crosses below Kijun
            tk_bearish_cross = curr_tenkan < curr_kijun and (i == start_idx or tenkan_aligned[i-1] >= kijun_aligned[i-1])
            
            # Volume confirmation: current volume > 1.5 * 20-period average volume
            # Note: vol_ratio_1w_aligned contains the ratio, so we check if > 1.5
            volume_confirmation = vol_ratio_1w_aligned[i] > 1.5
            
            # Long: Bullish TK cross AND price above cloud AND volume confirmation
            if tk_bullish_cross and curr_close > cloud_top and volume_confirmation:
                signals[i] = 0.25
                position = 1
            # Short: Bearish TK cross AND price below cloud AND volume confirmation
            elif tk_bearish_cross and curr_close < cloud_bottom and volume_confirmation:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_1dTKCross_VolumeConfirmation_v1"
timeframe = "6h"
leverage = 1.0