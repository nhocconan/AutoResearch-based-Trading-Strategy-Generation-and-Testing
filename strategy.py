#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud breakout with 1w EMA200 trend filter and volume confirmation.
# Targets 6h timeframe with ~12-37 trades/year. Long when price breaks above Kumo cloud with Tenkan > Kijun and volume spike.
# Short when price breaks below Kumo cloud with Tenkan < Kijun and volume spike.
# 1w EMA200 filter ensures alignment with major trend to avoid counter-trend whipsaws.
# Volume spike (>2.0x 20-bar average) confirms breakout strength.
# Position size 0.25 for balance between return and drawdown control.
# Discrete levels minimize fee churn. Works in both bull and bear via trend filter.

name = "6h_Ichimoku_1wEMA200_Trend_VolumeSpike_v1"
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
    
    # Get 1w data for EMA200 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA200 for trend filter
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
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
    
    # Kumo (Cloud) boundaries: Senkou Span A and B
    upper_cloud = np.maximum(senkou_a, senkou_b)
    lower_cloud = np.minimum(senkou_a, senkou_b)
    
    # Calculate 6h volume spike: >2.0x 20-bar average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 52  # Ensure sufficient history for Ichimoku (52-period)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_200_1w_aligned[i]) or 
            np.isnan(tenkan[i]) or 
            np.isnan(kijun[i]) or 
            np.isnan(upper_cloud[i]) or 
            np.isnan(lower_cloud[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price relative to 1w EMA200
        price_above_ema = close[i] > ema_200_1w_aligned[i]
        price_below_ema = close[i] < ema_200_1w_aligned[i]
        
        # Ichimoku breakout conditions with volume confirmation
        # Long: price breaks above cloud AND Tenkan > Kijun (bullish momentum)
        long_breakout = close[i] > upper_cloud[i] and tenkan[i] > kijun[i] and volume_spike[i]
        # Short: price breaks below cloud AND Tenkan < Kijun (bearish momentum)
        short_breakout = close[i] < lower_cloud[i] and tenkan[i] < kijun[i] and volume_spike[i]
        
        # Exit conditions: price returns to cloud or trend reversal
        long_exit = close[i] < lower_cloud[i] or close[i] < ema_200_1w_aligned[i]
        short_exit = close[i] > upper_cloud[i] or close[i] > ema_200_1w_aligned[i]
        
        # Handle entries and exits
        if long_breakout and price_above_ema and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_breakout and price_below_ema and position >= 0:
            signals[i] = -0.25
            position = -1
        elif (position == 1 and long_exit) or (position == -1 and short_exit):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals