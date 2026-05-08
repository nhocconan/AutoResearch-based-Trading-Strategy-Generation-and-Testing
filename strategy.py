#/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Ichimoku_Cloud_Breakout_1dTrend"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data once for Ichimoku and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    # Ichimoku components on daily (standard periods)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Align Ichimoku components to 6h timeframe
    tenkan_6h = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_6h = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_6h = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_6h = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # Cloud top and bottom
    cloud_top = np.maximum(senkou_a_6h, senkou_b_6h)
    cloud_bottom = np.minimum(senkou_a_6h, senkou_b_6h)
    
    # Trend filter: price above/below cloud on 1d
    price_above_cloud = (close > cloud_top).astype(float)
    price_below_cloud = (close < cloud_bottom).astype(float)
    trend_filter = align_htf_to_ltf(prices, df_1d, price_above_cloud - price_below_cloud)
    
    # Volume confirmation: current volume > 1.5 * 20-period average on 6h
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 52  # warmup for Ichimoku (max period 52)
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i]) or 
            np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]) or 
            np.isnan(trend_filter[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: TK cross bullish, price above cloud, volume spike, 1d trend bullish
            tk_bullish = tenkan_6h[i] > kijun_6h[i]
            price_above = close[i] > cloud_top[i]
            trend_bullish = trend_filter[i] > 0.5
            
            if tk_bullish and price_above and vol_spike[i] and trend_bullish:
                signals[i] = 0.25
                position = 1
            # Short entry: TK cross bearish, price below cloud, volume spike, 1d trend bearish
            elif (tenkan_6h[i] < kijun_6h[i] and 
                  close[i] < cloud_bottom[i] and 
                  vol_spike[i] and 
                  trend_filter[i] < -0.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: TK cross bearish OR price breaks below cloud bottom
            if (tenkan_6h[i] < kijun_6h[i] or close[i] < cloud_bottom[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: TK cross bullish OR price breaks above cloud top
            if (tenkan_6h[i] > kijun_6h[i] or close[i] > cloud_top[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Ichimoku TK cross with cloud filter and volume confirmation on 6h timeframe.
# Uses 1d Ichimoku for trend context and cloud boundaries. Works in bull markets (TK bullish + price above cloud) 
# and bear markets (TK bearish + price below cloud). Volume spike filters weak breakouts.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee decay.
# Cloud acts as dynamic support/resistance, TK cross captures momentum shifts.