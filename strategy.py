#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud with 12h EMA50 trend filter and volume confirmation
# Uses Ichimoku (Tenkan/Kijun cross + price vs Cloud) for trend/momentum signals.
# 12h EMA50 ensures alignment with medium-term trend to avoid counter-trend whipsaws.
# Volume spike confirms institutional participation. Works in bull/bear via trend filter.
# Target: 50-150 total trades over 4 years (12-37/year) with discrete sizing 0.25.

name = "6h_Ichimoku_12hEMA50_Trend_Volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate EMA(50) on 12h for trend filter
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Ichimoku components on 6h
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
    
    # Chikou Span (Lagging Span): close plotted 26 periods behind (not used for signals)
    
    # Determine cloud (Kumo) - future cloud is Senkou A/B shifted forward 26 periods
    # For signal detection, we use current cloud (Senkou A/B from 26 periods ago)
    senkou_a_lag = np.roll(senkou_a, 26)
    senkou_b_lag = np.roll(senkou_b, 26)
    senkou_a_lag[:26] = np.nan
    senkou_b_lag[:26] = np.nan
    
    # Cloud top and bottom
    cloud_top = np.maximum(senkou_a_lag, senkou_b_lag)
    cloud_bottom = np.minimum(senkou_a_lag, senkou_b_lag)
    
    # Volume confirmation (2.0x 20-period average) on 6h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for calculations)
    start_idx = 60
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(tenkan[i]) or np.isnan(kijun[i]) or
            np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Tenkan crosses above Kijun + price above cloud + 12h uptrend + volume spike
            if (tenkan[i] > kijun[i] and tenkan[i-1] <= kijun[i-1] and 
                close[i] > cloud_top[i] and close[i] > ema_50_12h_aligned[i] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: Tenkan crosses below Kijun + price below cloud + 12h downtrend + volume spike
            elif (tenkan[i] < kijun[i] and tenkan[i-1] >= kijun[i-1] and 
                  close[i] < cloud_bottom[i] and close[i] < ema_50_12h_aligned[i] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Tenkan crosses below Kijun or price falls below cloud or trend reversal
            if (tenkan[i] < kijun[i] and tenkan[i-1] >= kijun[i-1]) or close[i] < cloud_bottom[i] or close[i] < ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Tenkan crosses above Kijun or price rises above cloud or trend reversal
            if (tenkan[i] > kijun[i] and tenkan[i-1] <= kijun[i-1]) or close[i] > cloud_top[i] or close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals