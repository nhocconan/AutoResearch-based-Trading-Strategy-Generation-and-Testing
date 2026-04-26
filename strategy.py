#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Breakout_1wTrend_VolumeConfirmation
Hypothesis: 6-hour Ichimoku TK cross with 1-week cloud filter and volume confirmation.
In bull: price above cloud + TK cross up + volume spike → long.
In bear: price below cloud + TK cross down + volume spike → short.
Weekly trend (price vs weekly EMA20) filters direction to avoid counter-trend whipsaws.
Ichimoku works well on 6h timeframe for capturing medium-term trends with clear entry/exit.
Target: 12-30 trades/year to stay within fee limits.
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
    
    # Get 1w data for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly EMA(20) for trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = (period52_high + period52_low) / 2
    
    # Chikou Span (Lagging Span): close plotted 26 periods behind
    # Not used for signals due to look-ahead concerns
    
    # Calculate volume ratio (current / 20-period average) for spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.maximum(vol_ma, 1e-10)  # avoid division by zero
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: max of Ichimoku periods (52), weekly EMA(20), volume MA(20)
    start_idx = max(52, 20, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or
            np.isnan(senkou_a[i]) or np.isnan(senkou_b[i]) or
            np.isnan(ema_20_1w_aligned[i]) or np.isnan(vol_ratio[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        close_val = close[i]
        vol_spike = vol_ratio[i] > 2.0  # volume at least 2x average
        
        # Ichimoku signals
        tk_cross_up = tenkan[i] > kijun[i] and tenkan[i-1] <= kijun[i-1]
        tk_cross_down = tenkan[i] < kijun[i] and tenkan[i-1] >= kijun[i-1]
        
        # Cloud trend: price above/both Senkou lines
        cloud_top = max(senkou_a[i], senkou_b[i])
        cloud_bottom = min(senkou_a[i], senkou_b[i])
        price_above_cloud = close_val > cloud_top
        price_below_cloud = close_val < cloud_bottom
        
        # Weekly trend filter
        weekly_uptrend = close_val > ema_20_1w_aligned[i]
        weekly_downtrend = close_val < ema_20_1w_aligned[i]
        
        if position == 0:
            # Long: TK cross up + price above cloud + weekly uptrend + volume spike
            long_signal = tk_cross_up and price_above_cloud and weekly_uptrend and vol_spike
            
            # Short: TK cross down + price below cloud + weekly downtrend + volume spike
            short_signal = tk_cross_down and price_below_cloud and weekly_downtrend and vol_spike
            
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
            # Exit: TK cross down OR price falls below cloud OR weekly trend turns down
            if tk_cross_down or (close_val < cloud_top) or (not weekly_uptrend):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: TK cross up OR price rises above cloud OR weekly trend turns up
            if tk_cross_up or (close_val > cloud_bottom) or (not weekly_downtrend):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_Cloud_Breakout_1wTrend_VolumeConfirmation"
timeframe = "6h"
leverage = 1.0