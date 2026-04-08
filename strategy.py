#!/usr/bin/env python3
# 6h_ichimoku_cloud_1d_trend_volume_v1
# Hypothesis: Ichimoku cloud from daily timeframe provides trend direction and support/resistance,
# while Tenkan-Kijun cross on 6h gives entry signals with volume confirmation.
# Works in bull/bear by following higher timeframe trend (cloud color) and using
# momentum crosses for entries. Targets 50-150 total trades over 4 years.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ichimoku_cloud_1d_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Ichimoku
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Chikou Span (Lagging Span): close shifted 26 periods behind
    # Not used for signals to avoid look-ahead
    
    # Align Ichimoku components to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # Determine cloud color (trend direction)
    # Green bullish cloud when Senkou A > Senkou B
    # Red bearish cloud when Senkou A < Senkou B
    cloud_green = senkou_a_aligned > senkou_b_aligned
    
    # Tenkan-Kijun cross on 6h
    # Tenkan-sen (9-period) on 6h
    tenkan_sen_6h = (pd.Series(high).rolling(window=9, min_periods=9).max().values + 
                     pd.Series(low).rolling(window=9, min_periods=9).min().values) / 2
    # Kijun-sen (26-period) on 6h
    kijun_sen_6h = (pd.Series(high).rolling(window=26, min_periods=26).max().values + 
                    pd.Series(low).rolling(window=26, min_periods=26).min().values) / 2
    
    # TK cross signals
    tk_cross_up = (tenkan_sen_6h > kijun_sen_6h) & (np.roll(tenkan_sen_6h, 1) <= np.roll(kijun_sen_6h, 1))
    tk_cross_down = (tenkan_sen_6h < kijun_sen_6h) & (np.roll(tenkan_sen_6h, 1) >= np.roll(kijun_sen_6h, 1))
    
    # Volume filter: 6h volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup (need 52 periods for Ichimoku)
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(tenkan_sen_6h[i]) or np.isnan(kijun_sen_6h[i]) or 
            np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or
            np.isnan(volume_filter[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: TK cross down OR price enters cloud from above (cloud acts as support/resistance)
            if tk_cross_down[i] or (
                (close[i] < senkou_a_aligned[i] and close[i] < senkou_b_aligned[i]) or
                (close[i] > senkou_a_aligned[i] and close[i] > senkou_b_aligned[i])
            ):
                # Check if we've crossed into the cloud (trend change)
                if (close[i] >= senkou_a_aligned[i] and close[i] >= senkou_b_aligned[i]) or \
                   (close[i] <= senkou_a_aligned[i] and close[i] <= senkou_b_aligned[i]):
                    price_in_cloud = True
                else:
                    price_in_cloud = False
                
                if tk_cross_down[i] or price_in_cloud:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: TK cross up OR price enters cloud from below
            if tk_cross_up[i] or (
                (close[i] < senkou_a_aligned[i] and close[i] < senkou_b_aligned[i]) or
                (close[i] > senkou_a_aligned[i] and close[i] > senkou_b_aligned[i])
            ):
                # Check if we've crossed into the cloud (trend change)
                if (close[i] >= senkou_a_aligned[i] and close[i] >= senkou_b_aligned[i]) or \
                   (close[i] <= senkou_a_aligned[i] and close[i] <= senkou_b_aligned[i]):
                    price_in_cloud = True
                else:
                    price_in_cloud = False
                
                if tk_cross_up[i] or price_in_cloud:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: TK cross up + price above cloud (bullish) + volume
            if (tk_cross_up[i] and 
                close[i] > senkou_a_aligned[i] and close[i] > senkou_b_aligned[i] and
                cloud_green[i] and volume_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: TK cross down + price below cloud (bearish) + volume
            elif (tk_cross_down[i] and 
                  close[i] < senkou_a_aligned[i] and close[i] < senkou_b_aligned[i] and
                  not cloud_green[i] and volume_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals