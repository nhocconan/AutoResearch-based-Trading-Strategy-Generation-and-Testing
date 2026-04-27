#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_TK_Cross_1dTrend
Hypothesis: Ichimoku TK cross with 1d trend filter on 6h timeframe. Uses Tenkan/Kijun cross for entry timing, 
1d EMA50 for trend direction, and cloud (Senkou Span) as dynamic support/resistance. 
Designed for 6h targeting 50-150 trades over 4 years (12-37/year). Works in bull/bear: 
In uptrends (price > 1d EMA50), long on TK cross above cloud; in downtrends (price < 1d EMA50), 
short on TK cross below cloud. Exit on reverse TK cross or trend reversal.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Current cloud boundaries (Senkou Span A/B shifted back 26 periods to align with current price)
    # The cloud plotted today is actually Senkou Span A/B calculated 26 periods ago
    senkou_a_lagged = np.roll(senkou_a, 26)
    senkou_b_lagged = np.roll(senkou_b, 26)
    # First 26 values are invalid due to roll
    senkou_a_lagged[:26] = np.nan
    senkou_b_lagged[:26] = np.nan
    
    # Cloud top and bottom
    cloud_top = np.maximum(senkou_a_lagged, senkou_b_lagged)
    cloud_bottom = np.minimum(senkou_a_lagged, senkou_b_lagged)
    
    # 1d EMA50 trend filter
    ema_50 = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25  # Position size to balance return and drawdown
    
    # Warmup: need Ichimoku (52+26=78) and 1d EMA
    start_idx = max(80, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]) or
            np.isnan(ema_50_aligned[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        tenkan_val = tenkan[i]
        kijun_val = kijun[i]
        cloud_top_val = cloud_top[i]
        cloud_bottom_val = cloud_bottom[i]
        ema_val = ema_50_aligned[i]
        
        # TK cross signals
        tk_cross_up = tenkan_val > kijun_val and tenkan[i-1] <= kijun[i-1]
        tk_cross_down = tenkan_val < kijun_val and tenkan[i-1] >= kijun[i-1]
        
        if position == 0:
            # Look for entry: TK cross in direction of 1d trend, price outside cloud
            if ema_val is not None and not np.isnan(ema_val):
                # Uptrend: price > 1d EMA50
                if close_val > ema_val:
                    # Long: TK cross up AND price above cloud (strong bullish)
                    if tk_cross_up and close_val > cloud_top_val:
                        signals[i] = size
                        position = 1
                # Downtrend: price < 1d EMA50
                elif close_val < ema_val:
                    # Short: TK cross down AND price below cloud (strong bearish)
                    if tk_cross_down and close_val < cloud_bottom_val:
                        signals[i] = -size
                        position = -1
        elif position == 1:
            # Exit long: TK cross down OR price re-enters cloud OR trend reversal
            if tk_cross_down or close_val < cloud_bottom_val or close_val < ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: TK cross up OR price re-enters cloud OR trend reversal
            if tk_cross_up or close_val > cloud_top_val or close_val > ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Ichimoku_Cloud_TK_Cross_1dTrend"
timeframe = "6h"
leverage = 1.0