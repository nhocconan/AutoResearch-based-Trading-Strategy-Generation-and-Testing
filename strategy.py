#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud with 1d filter
# Long when price above Kumo (cloud) AND Tenkan > Kijun (bullish TK cross) AND 1d close > 1d Kumo top
# Short when price below Kumo AND Tenkan < Kijun (bearish TK cross) AND 1d close < 1d Kumo bottom
# Exit when TK cross reverses or price crosses Kijun
# Uses 6h timeframe for balanced trade frequency, Ichimoku for trend/momentum, 1d for higher timeframe filter
# Target: 50-150 total trades over 4 years (12-37/year)

name = "6h_ichimoku_1d_filter_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Ichimoku Cloud components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max()
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min()
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max()
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min()
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max()
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min()
    senkou_b = (period52_high + period52_low) / 2
    
    # 1-day Ichimoku filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d Ichimoku components
    period9_high_1d = pd.Series(high_1d).rolling(window=9, min_periods=9).max()
    period9_low_1d = pd.Series(low_1d).rolling(window=9, min_periods=9).min()
    tenkan_1d = (period9_high_1d + period9_low_1d) / 2
    
    period26_high_1d = pd.Series(high_1d).rolling(window=26, min_periods=26).max()
    period26_low_1d = pd.Series(low_1d).rolling(window=26, min_periods=26).min()
    kijun_1d = (period26_high_1d + period26_low_1d) / 2
    
    senkou_a_1d = (tenkan_1d + kijun_1d) / 2
    period52_high_1d = pd.Series(high_1d).rolling(window=52, min_periods=52).max()
    period52_low_1d = pd.Series(low_1d).rolling(window=52, min_periods=52).min()
    senkou_b_1d = (period52_high_1d + period52_low_1d) / 2
    
    # 1d Kumo (cloud) boundaries
    kumO_top_1d = np.maximum(senkou_a_1d, senkou_b_1d)
    kumO_bottom_1d = np.minimum(senkou_a_1d, senkou_b_1d)
    
    # Align 1d Ichimoku to 6h timeframe
    kumO_top_1d_aligned = align_htf_to_ltf(prices, df_1d, kumO_top_1d)
    kumO_bottom_1d_aligned = align_htf_to_ltf(prices, df_1d, kumO_bottom_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(52, n):  # Start after 52 periods for full Ichimoku
        # Skip if required data not available
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or np.isnan(senkou_a[i]) or 
            np.isnan(senkou_b[i]) or np.isnan(kumO_top_1d_aligned[i]) or 
            np.isnan(kumO_bottom_1d_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Current 6h Kumo (cloud) boundaries
        kumO_top = np.maximum(senkou_a[i], senkou_b[i])
        kumO_bottom = np.minimum(senkou_a[i], senkou_b[i])
        
        # TK cross conditions
        tk_bullish = tenkan[i] > kijun[i]
        tk_bearish = tenkan[i] < kijun[i]
        
        # Price relative to Kumo
        price_above_kumo = close[i] > kumO_top
        price_below_kumo = close[i] < kumO_bottom
        
        # 1d filter: close relative to 1d Kumo
        price_above_1d_kumo = close_1d[i] > kumO_top_1d_aligned[i] if i < len(close_1d) else False
        price_below_1d_kumo = close_1d[i] < kumO_bottom_1d_aligned[i] if i < len(close_1d) else False
        
        if position == 1:  # long position
            # Exit: TK cross turns bearish OR price drops below Kijun
            if tk_bearish or close[i] < kijun[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: TK cross turns bullish OR price rises above Kijun
            if tk_bullish or close[i] > kijun[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries
            # Long: price above Kumo AND bullish TK cross AND 1d price above 1d Kumo
            if price_above_kumo and tk_bullish and price_above_1d_kumo:
                signals[i] = 0.25
                position = 1
            # Short: price below Kumo AND bearish TK cross AND 1d price below 1d Kumo
            elif price_below_kumo and tk_bearish and price_below_1d_kumo:
                signals[i] = -0.25
                position = -1
    
    return signals