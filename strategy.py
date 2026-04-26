#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Filter_1dTrend_VolumeBreakout_v1
Hypothesis: Ichimoku cloud (TK cross + price outside cloud) on 6h with 1d EMA50 trend filter and 2x volume breakout confirmation. 
Works in bull markets via breakout above cloud with trend alignment; works in bear markets via breakdown below cloud with trend alignment. 
Ichimoku cloud acts as dynamic support/resistance, reducing whipsaw. Volume breakout confirms institutional participation. 
Target: 15-35 trades/year (60-140 total over 4 years) to minimize fee drag. Discrete sizing 0.25.
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
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate EMA(50) on 1d for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Ichimoku components on 6h (primary timeframe)
    # Conversion line (Tenkan-sen): (9-period high + low)/2
    period_9 = 9
    high_9 = pd.Series(high).rolling(window=period_9, min_periods=period_9).max().values
    low_9 = pd.Series(low).rolling(window=period_9, min_periods=period_9).min().values
    tenkan = (high_9 + low_9) / 2.0
    
    # Base line (Kijun-sen): (26-period high + low)/2
    period_26 = 26
    high_26 = pd.Series(high).rolling(window=period_26, min_periods=period_26).max().values
    low_26 = pd.Series(low).rolling(window=period_26, min_periods=period_26).min().values
    kijun = (high_26 + low_26) / 2.0
    
    # Leading Span A (Senkou Span A): (Tenkan + Kijun)/2 plotted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2.0)
    
    # Leading Span B (Senkou Span B): (52-period high + low)/2 plotted 26 periods ahead
    period_52 = 52
    high_52 = pd.Series(high).rolling(window=period_52, min_periods=period_52).max().values
    low_52 = pd.Series(low).rolling(window=period_52, min_periods=period_52).min().values
    senkou_b = ((high_52 + low_52) / 2.0)
    
    # Align Ichimoku components to current bar (no look-ahead)
    # Since Senkou spans are plotted ahead, we need to shift them back for current cloud
    tenkan_aligned = align_htf_to_ltf(prices, prices, tenkan)  # same timeframe, no shift needed
    kijun_aligned = align_htf_to_ltf(prices, prices, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, prices, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, prices, senkou_b)
    
    # Volume breakout: volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_breakout = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: max of Ichimoku periods, EMA, volume MA
    start_idx = max(52, 50, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_aligned[i]) or
            np.isnan(kijun_aligned[i]) or
            np.isnan(senkou_a_aligned[i]) or
            np.isnan(senkou_b_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(vol_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        close_val = close[i]
        # Cloud boundaries (Senkou Span A and B)
        upper_cloud = max(senkou_a_aligned[i], senkou_b_aligned[i])
        lower_cloud = min(senkou_a_aligned[i], senkou_b_aligned[i])
        # Trend filter
        trend_1d_up = close_val > ema_50_1d_aligned[i]
        trend_1d_down = close_val < ema_50_1d_aligned[i]
        vol_break = volume_breakout[i]
        
        if position == 0:
            # Long: price breaks above cloud AND TK cross bullish AND 1d trend up AND volume breakout
            tk_bullish = tenkan_aligned[i] > kijun_aligned[i]
            long_signal = (close_val > upper_cloud) and tk_bullish and trend_1d_up and vol_break
            
            # Short: price breaks below cloud AND TK cross bearish AND 1d trend down AND volume breakout
            tk_bearish = tenkan_aligned[i] < kijun_aligned[i]
            short_signal = (close_val < lower_cloud) and tk_bearish and trend_1d_down and vol_break
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price re-enters cloud OR TK cross bearish OR trend flips down
            if (close_val < lower_cloud) or (tenkan_aligned[i] < kijun_aligned[i]) or (not trend_1d_up):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price re-enters cloud OR TK cross bullish OR trend flips up
            if (close_val > upper_cloud) or (tenkan_aligned[i] > kijun_aligned[i]) or (not trend_1d_down):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_Cloud_Filter_1dTrend_VolumeBreakout_v1"
timeframe = "6h"
leverage = 1.0