#!/usr/bin/env python3
"""
6h_Ichimoku_TK_Cross_Cloud_Filter_12hTrend
Hypothesis: Ichimoku TK cross (Tenkan/Kijun) with price above/below cloud (from 1d) for trend confirmation, filtered by 12h EMA50 trend direction. Uses discrete sizing (0.25) to limit fee churn. Works in bull/bear via 12h trend filter and cloud as dynamic support/resistance. Targets 50-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:  # Need warmup for Ichimoku (52) and EMA
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data for HTF trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 60:
        return np.zeros(n)
    
    # 12h EMA50 for trend filter
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
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
    
    # Current cloud boundaries (Senkou Span A/B shifted back 26 periods to align with price)
    # We need values from 26 periods ago to represent current cloud
    senkou_a_lagged = np.roll(senkou_a, 26)
    senkou_b_lagged = np.roll(senkou_b, 26)
    # First 26 values are invalid due to roll
    senkou_a_lagged[:26] = np.nan
    senkou_b_lagged[:26] = np.nan
    
    # Cloud top/bottom
    cloud_top = np.maximum(senkou_a_lagged, senkou_b_lagged)
    cloud_bottom = np.minimum(senkou_a_lagged, senkou_b_lagged)
    
    # TK Cross signals
    tk_cross_up = (tenkan > kijun) & (np.roll(tenkan, 1) <= np.roll(kijun, 1))  # Tenkan crosses above Kijun
    tk_cross_down = (tenkan < kijun) & (np.roll(tenkan, 1) >= np.roll(kijun, 1))  # Tenkan crosses below Kijun
    
    # Price relative to cloud
    price_above_cloud = close > cloud_top
    price_below_cloud = close < cloud_bottom
    
    # Volume confirmation: current volume > 1.5x average volume (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (1.5 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Start after warmup (need 52 for Ichimoku, 50 for EMA, 20 for volume)
    start_idx = max(52, 50, 20) + 26  # +26 for cloud lag
    
    for i in range(start_idx, n):
        # Get current values
        tk_up = tk_cross_up[i]
        tk_down = tk_cross_down[i]
        price_above = price_above_cloud[i]
        price_below = price_below_cloud[i]
        ema_val = ema_50_12h_aligned[i]
        vol_ok = volume_confirmed[i]
        
        # Skip if data not ready
        if np.isnan(ema_val) or np.isnan(avg_volume[i]):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Long: TK cross up + price above cloud + 12h uptrend + volume
        long_condition = tk_up and price_above and (close[i] > ema_val) and vol_ok
        # Short: TK cross down + price below cloud + 12h downtrend + volume
        short_condition = tk_down and price_below and (close[i] < ema_val) and vol_ok
        
        # Exit: opposite TK cross OR price crosses cloud in opposite direction
        exit_long = tk_down or (close[i] < cloud_bottom[i])
        exit_short = tk_up or (close[i] > cloud_top[i])
        
        if long_condition and position != 1:
            signals[i] = base_size
            position = 1
        elif short_condition and position != -1:
            signals[i] = -base_size
            position = -1
        elif position == 1 and exit_long:
            signals[i] = 0.0
            position = 0
        elif position == -1 and exit_short:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "6h_Ichimoku_TK_Cross_Cloud_Filter_12hTrend"
timeframe = "6h"
leverage = 1.0