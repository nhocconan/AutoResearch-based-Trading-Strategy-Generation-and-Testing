#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud breakout with 1d trend filter and volume confirmation
# Uses Ichimoku (Tenkan, Kijun, Senkou Span A/B) on 6h for entry signals.
# Long when price breaks above cloud with bullish TK cross and 1d uptrend.
# Short when price breaks below cloud with bearish TK cross and 1d downtrend.
# Volume confirmation filters weak breakouts. Designed for 12-37 trades/year on 6h.

name = "6h_Ichimoku_Cloud_Breakout_1dTrend_VolumeConfirmation"
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
    
    # Get 1d data for trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Ichimoku components (9, 26, 52 periods) on 6h data
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
    
    # Align Ichimoku components to avoid look-ahead (Senkou spans already shifted)
    # For entry, we use current Tenkan/Kijun and current Senkou A/B (which represent future cloud)
    tenkan_aligned = align_htf_to_ltf(prices, prices, tenkan)  # self-align for proper indexing
    kijun_aligned = align_htf_to_ltf(prices, prices, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, prices, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, prices, senkou_b)
    
    # Calculate volume spike filter (20-period volume EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ema_20 * 2.0)  # Volume at least 2x average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(52, n):  # Need 52 periods for Senkou B
        # Skip if any value is NaN
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Cloud top and bottom
        cloud_top = max(senkou_a_aligned[i], senkou_b_aligned[i])
        cloud_bottom = min(senkou_a_aligned[i], senkou_b_aligned[i])
        
        if position == 0:
            # Long conditions: price breaks above cloud AND bullish TK cross AND 1d uptrend AND volume spike
            if (close[i] > cloud_top and 
                tenkan_aligned[i] > kijun_aligned[i] and  # bullish TK cross
                close[i] > ema50_1d_aligned[i] and  # 1d uptrend filter
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below cloud AND bearish TK cross AND 1d downtrend AND volume spike
            elif (close[i] < cloud_bottom and 
                  tenkan_aligned[i] < kijun_aligned[i] and  # bearish TK cross
                  close[i] < ema50_1d_aligned[i] and  # 1d downtrend filter
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below cloud OR bearish TK cross OR trend reverses
            if (close[i] < cloud_top or 
                tenkan_aligned[i] < kijun_aligned[i] or
                close[i] < ema50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above cloud OR bullish TK cross OR trend reverses
            if (close[i] > cloud_bottom or 
                tenkan_aligned[i] > kijun_aligned[i] or
                close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals