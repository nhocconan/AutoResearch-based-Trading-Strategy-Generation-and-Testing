#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud with 1d trend filter and volume confirmation.
# Long when Tenkan-sen crosses above Kijun-sen AND price above Kumo cloud (Senkou Span A/B) 
# AND 1d EMA50 uptrend AND volume > 1.5x 20-period average.
# Short when Tenkan-sen crosses below Kijun-sen AND price below Kumo cloud 
# AND 1d EMA50 downtrend AND volume > 1.5x 20-period average.
# Exit when Tenkan-sen/Kijun-sen cross reverses OR price crosses Kumo cloud in opposite direction.
# Uses Ichimoku for trend/momentum with multi-timeframe alignment to avoid look-ahead.
# Target: 80-160 total trades over 4 years (20-40/year) for balanced freq/performance.

name = "6h_Ichimoku_1dEMA50_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    # 6h volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 52  # Sufficient warmup for Ichimoku (need 52 for Senkou B)
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(senkou_a[i]) or np.isnan(senkou_b[i]) or
            np.isnan(volume_filter[i]) or np.isnan(ema50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine cloud top and bottom (Senkou Span A/B)
        cloud_top = max(senkou_a[i], senkou_b[i])
        cloud_bottom = min(senkou_a[i], senkou_b[i])
        
        # Price relative to cloud
        price_above_cloud = close[i] > cloud_top
        price_below_cloud = close[i] < cloud_bottom
        
        if position == 0:
            # Long conditions: TK cross up, price above cloud, 1d uptrend, volume spike
            tk_cross_up = (tenkan[i] > kijun[i]) and (tenkan[i-1] <= kijun[i-1])
            long_cond = tk_cross_up and price_above_cloud and (close[i] > ema50_1d_aligned[i]) and volume_filter[i]
            
            # Short conditions: TK cross down, price below cloud, 1d downtrend, volume spike
            tk_cross_down = (tenkan[i] < kijun[i]) and (tenkan[i-1] >= kijun[i-1])
            short_cond = tk_cross_down and price_below_cloud and (close[i] < ema50_1d_aligned[i]) and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: TK cross down OR price drops below cloud
            tk_cross_down = (tenkan[i] < kijun[i]) and (tenkan[i-1] >= kijun[i-1])
            price_below_cloud = close[i] < cloud_bottom
            if tk_cross_down or price_below_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: TK cross up OR price rises above cloud
            tk_cross_up = (tenkan[i] > kijun[i]) and (tenkan[i-1] <= kijun[i-1])
            price_above_cloud = close[i] > cloud_top
            if tk_cross_up or price_above_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals