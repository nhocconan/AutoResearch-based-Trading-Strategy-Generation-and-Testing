#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud breakout with 1d trend filter and volume confirmation
# Ichimoku provides dynamic support/resistance (cloud) and momentum (TK cross)
# 1d EMA50 ensures alignment with daily trend to avoid counter-trend trades
# Volume spike (>1.8x 20-period EMA) filters low-probability breakouts
# Target: 80-150 total trades over 4 years (20-37/year) to balance edge and fee drag

name = "6h_Ichimoku_CloudBreakout_1dEMA50_VolumeSpike"
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
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Ichimoku Cloud calculation (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    high_9 = pd.Series(high).rolling(window=9, min_periods=9).max().values
    low_9 = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (high_9 + low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    high_26 = pd.Series(high).rolling(window=26, min_periods=26).max().values
    low_26 = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (high_26 + low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    high_52 = pd.Series(high).rolling(window=52, min_periods=52).max().values
    low_52 = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = (high_52 + low_52) / 2
    
    # The cloud is between Senkou Span A and Senkou Span B
    # For bullish cloud: Senkou A > Senkou B
    # For bearish cloud: Senkou A < Senkou B
    
    # Volume confirmation: 20-period EMA on 6h volume
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(52, n):  # Start from 52 to have valid Ichimoku
        # Skip if any value is NaN
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(senkou_a[i]) or np.isnan(senkou_b[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 1.8 x 20-period EMA (tight to avoid overtrading)
        volume_spike = volume[i] > (1.8 * vol_ema_20[i])
        
        # Ichimoku signals with 1d trend filter
        # Bullish: Price above cloud + TK cross bullish + price above 1d EMA50 + volume spike
        # Bearish: Price below cloud + TK cross bearish + price below 1d EMA50 + volume spike
        if position == 0:
            bullish_condition = (close[i] > max(senkou_a[i], senkou_b[i]) and  # Price above cloud
                                tenkan[i] > kijun[i] and                    # TK cross bullish
                                close[i] > ema_50_1d_aligned[i] and         # Above 1d EMA50
                                volume_spike)                               # Volume confirmation
            
            bearish_condition = (close[i] < min(senkou_a[i], senkou_b[i]) and  # Price below cloud
                                tenkan[i] < kijun[i] and                    # TK cross bearish
                                close[i] < ema_50_1d_aligned[i] and         # Below 1d EMA50
                                volume_spike)                               # Volume confirmation
            
            if bullish_condition:
                signals[i] = 0.25
                position = 1
            elif bearish_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price falls below cloud OR TK cross turns bearish
            if close[i] < min(senkou_a[i], senkou_b[i]) or tenkan[i] < kijun[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price rises above cloud OR TK cross turns bullish
            if close[i] > max(senkou_a[i], senkou_b[i]) or tenkan[i] > kijun[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals