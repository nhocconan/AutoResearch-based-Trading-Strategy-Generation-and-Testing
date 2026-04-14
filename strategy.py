#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with 1d trend filter and volume confirmation
# Trend from 1d EMA (50) provides directional bias to avoid counter-trend trades
# 12h Camarilla pivot levels (H3/L3) breakouts capture momentum in direction of 1d trend
# Volume > 1.5x average confirms institutional participation
# Works in bull/bear as 1d EMA adapts to trend
# Target: 15-30 trades/year per symbol (60-120 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA(50) for trend filter
    ema_len = 50
    if len(df_1d) < ema_len:
        return np.zeros(n)
    
    ema_1d = pd.Series(df_1d['close']).ewm(span=ema_len, adjust=False, min_periods=ema_len).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Load 1d data again for Camarilla pivot calculation (needs OHLC)
    # We'll use the same df_1d since it contains OHLC
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels for 1d timeframe
    # Camarilla levels based on previous day's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point and Camarilla levels
    pivot = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # Camarilla levels: H3/H4 and L3/L4
    # H3 = close + (high - low) * 1.1/2
    # L3 = close - (high - low) * 1.1/2
    camarilla_h3 = close_1d + range_1d * 1.1 / 2
    camarilla_l3 = close_1d - range_1d * 1.1 / 2
    
    # Align Camarilla levels to 12h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Volume confirmation: 1.5x average volume (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(60, 20)  # EMA50 needs 50, volume MA needs 20
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_1d_aligned[i]) or 
            np.isnan(camarilla_h3_aligned[i]) or
            np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price relative to 1d EMA50
        above_ema = close[i] > ema_1d_aligned[i]
        below_ema = close[i] < ema_1d_aligned[i]
        
        # Volume confirmation: current volume > 1.5x average
        volume_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Enter long: price breaks above Camarilla H3 + above 1d EMA + volume
            if (close[i] > camarilla_h3_aligned[i] and 
                above_ema and 
                volume_confirmed):
                position = 1
                signals[i] = position_size
            # Enter short: price breaks below Camarilla L3 + below 1d EMA + volume
            elif (close[i] < camarilla_l3_aligned[i] and 
                  below_ema and 
                  volume_confirmed):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to 1d EMA or breaks below Camarilla L3
            if close[i] < ema_1d_aligned[i] or close[i] < camarilla_l3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to 1d EMA or breaks above Camarilla H3
            if close[i] > ema_1d_aligned[i] or close[i] > camarilla_h3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1d_Camarilla_Pivot_EMA50_Volume_v1"
timeframe = "12h"
leverage = 1.0