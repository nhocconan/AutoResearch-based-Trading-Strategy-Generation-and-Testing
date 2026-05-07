#!/usr/bin/env python3
name = "6h_Ichimoku_TK_Cross_CloudFilter_1dTrend"
timeframe = "6h"
leverage = 1.0

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
    
    # Get 1d data for Ichimoku calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    # Ichimoku components on 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = np.full(len(high_1d), np.nan)
    period9_low = np.full(len(low_1d), np.nan)
    for i in range(9, len(high_1d)):
        period9_high[i] = np.max(high_1d[i-9:i+1])
        period9_low[i] = np.min(low_1d[i-9:i+1])
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = np.full(len(high_1d), np.nan)
    period26_low = np.full(len(low_1d), np.nan)
    for i in range(26, len(high_1d)):
        period26_high[i] = np.max(high_1d[i-26:i+1])
        period26_low[i] = np.min(low_1d[i-26:i+1])
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = np.full(len(high_1d), np.nan)
    period52_low = np.full(len(low_1d), np.nan)
    for i in range(52, len(high_1d)):
        period52_high[i] = np.max(high_1d[i-52:i+1])
        period52_low[i] = np.min(low_1d[i-52:i+1])
    senkou_b = (period52_high + period52_low) / 2
    
    # Align Ichimoku components to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # 12h trend filter for additional confirmation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume filter: current volume > 1.5x 20-period average (for 6h)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    vol_filter = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 2  # ~12 hours for 6h to reduce trades
    
    start_idx = max(100, 52)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_aligned[i]) or 
            np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or 
            np.isnan(senkou_b_aligned[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                bars_since_last_trade += 1
            continue
        
        bars_since_last_trade += 1
        
        # Ichimoku signals
        tk_cross_bull = tenkan_aligned[i] > kijun_aligned[i]
        tk_cross_bear = tenkan_aligned[i] < kijun_aligned[i]
        
        # Cloud: green when Senkou A > Senkou B, red when Senkou A < Senkou B
        cloud_green = senkou_a_aligned[i] > senkou_b_aligned[i]
        cloud_red = senkou_a_aligned[i] < senkou_b_aligned[i]
        
        # Price above/below cloud
        price_above_cloud = close[i] > max(senkou_a_aligned[i], senkou_b_aligned[i])
        price_below_cloud = close[i] < min(senkou_a_aligned[i], senkou_b_aligned[i])
        
        # Determine 12h trend direction
        trend_up = close > ema_50_12h_aligned[i]
        trend_down = close < ema_50_12h_aligned[i]
        
        if position == 0 and bars_since_last_trade >= cooldown_bars:
            # Long: TK cross bullish + price above green cloud + uptrend + volume
            if (tk_cross_bull and 
                price_above_cloud and 
                cloud_green and 
                trend_up[i] and 
                vol_filter[i]):
                signals[i] = 0.25
                position = 1
                bars_since_last_trade = 0
            # Short: TK cross bearish + price below red cloud + downtrend + volume
            elif (tk_cross_bear and 
                  price_below_cloud and 
                  cloud_red and 
                  trend_down[i] and 
                  vol_filter[i]):
                signals[i] = -0.25
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit: TK cross bearish or price falls below cloud
            if tk_cross_bear or not price_above_cloud:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: TK cross bullish or price rises above cloud
            if tk_cross_bull or not price_below_cloud:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Ichimoku TK cross with cloud filter and 12h EMA trend on 6h timeframe.
# Long when TK crosses bullish, price above green cloud, in uptrend with volume.
# Short when TK crosses bearish, price below red cloud, in downtrend with volume.
# Uses 1d Ichimoku for superior support/resistance and 12h EMA for trend filter.
# Works in bull markets (bullish TK cross in uptrend) and bear markets (bearish TK cross in downtrend).
# Target: 50-150 total trades over 4 years (12-37/year) as per experiment guidelines.