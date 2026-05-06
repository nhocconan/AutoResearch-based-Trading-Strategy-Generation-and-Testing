#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Ichimoku Cloud (Senkou Span A/B) with Tenkan/Kijun cross and volume confirmation
# Long when Tenkan crosses above Kijun AND price is above Cloud AND 1d EMA50 is rising AND volume > 1.5 * avg(20)
# Short when Tenkan crosses below Kijun AND price is below Cloud AND 1d EMA50 is falling AND volume > 1.5 * avg(20)
# Exit when Tenkan/Kijun cross reverses OR price touches opposite Cloud boundary
# Uses discrete sizing 0.25 to control drawdown and minimize fee churn
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe
# Ichimoku provides dynamic support/resistance with trend/momentum confirmation
# Works in bull (breakouts above Cloud) and bear (breakdowns below Cloud) via Cloud filter
# Volume confirmation ensures only high-conviction signals are taken

name = "6h_1dIchimoku_TK_Cross_CloudFilter_1dEMA50_Volume"
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
    
    # Get 1d data ONCE before loop for Ichimoku and EMA50
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # Need at least 52 completed daily bars for Ichimoku (26*2)
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Ichimoku components
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2.0
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2.0
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 plotted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2.0)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 plotted 26 periods ahead
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2.0)
    
    # Align Ichimoku components to 6h timeframe (wait for completed 1d bar)
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # Calculate 1d EMA50 trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate volume confirmation: volume > 1.5 * 20-period average volume on 6h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(avg_volume_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine Cloud boundaries (Senkou Span A/B form the Cloud)
        upper_cloud = np.maximum(senkou_a_aligned[i], senkou_b_aligned[i])
        lower_cloud = np.minimum(senkou_a_aligned[i], senkou_b_aligned[i])
        
        if position == 0:
            # Long: Tenkan crosses above Kijun AND price above Cloud AND EMA50 rising AND volume spike
            tenkan_cross_up = (tenkan_aligned[i] > kijun_aligned[i] and tenkan_aligned[i-1] <= kijun_aligned[i-1])
            price_above_cloud = close[i] > upper_cloud
            ema50_rising = ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1]
            if tenkan_cross_up and price_above_cloud and ema50_rising and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: Tenkan crosses below Kijun AND price below Cloud AND EMA50 falling AND volume spike
            elif (tenkan_aligned[i] < kijun_aligned[i] and tenkan_aligned[i-1] >= kijun_aligned[i-1] and
                  close[i] < lower_cloud and ema_50_1d_aligned[i] < ema_50_1d_aligned[i-1] and volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Tenkan/Kijun cross reverses OR price touches lower Cloud
            tenkan_cross_down = (tenkan_aligned[i] < kijun_aligned[i] and tenkan_aligned[i-1] >= kijun_aligned[i-1])
            price_touches_lower_cloud = close[i] <= lower_cloud
            if tenkan_cross_down or price_touches_lower_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Tenkan/Kijun cross reverses OR price touches upper Cloud
            tenkan_cross_up = (tenkan_aligned[i] > kijun_aligned[i] and tenkan_aligned[i-1] <= kijun_aligned[i-1])
            price_touches_upper_cloud = close[i] >= upper_cloud
            if tenkan_cross_up or price_touches_upper_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals