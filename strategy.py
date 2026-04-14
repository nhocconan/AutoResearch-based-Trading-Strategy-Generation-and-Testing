#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla L4 breakout with 1d trend filter (EMA200) and volume confirmation.
# Camarilla levels derived from 1d provide statistically significant support/resistance.
# Trend from 1d EMA200 provides directional bias to avoid counter-trend trades.
# Breakout of L4 (long) or H4 (short) with volume > 1.8x average confirms momentum.
# Works in bull/bear as 1d EMA200 adapts to trend.
# Target: 20-40 trades/year per symbol (80-160 total over 4 years).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for Camarilla and trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA(200) for trend filter
    ema_len = 200
    if len(df_1d) < ema_len:
        return np.zeros(n)
    
    ema_1d = pd.Series(df_1d['close']).ewm(span=ema_len, adjust=False, min_periods=ema_len).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate Camarilla levels from 1d OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla formula: Range = high - low
    range_1d = high_1d - low_1d
    # L4 = close + range * 1.12
    # H4 = close - range * 1.12
    camarilla_l4 = close_1d + range_1d * 1.12
    camarilla_h4 = close_1d - range_1d * 1.12
    
    # Align Camarilla levels to 4h timeframe
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    
    # Volume confirmation: 1.8x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(50, ema_len, 20)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(camarilla_l4_aligned[i]) or 
            np.isnan(camarilla_h4_aligned[i]) or
            np.isnan(ema_1d_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price relative to 1d EMA200
        above_ema = close[i] > ema_1d_aligned[i]
        below_ema = close[i] < ema_1d_aligned[i]
        
        # Volume confirmation: current volume > 1.8x average
        volume_confirmed = volume[i] > 1.8 * vol_ma[i]
        
        if position == 0:
            # Enter long: price breaks above Camarilla L4 + above 1d EMA200 + volume
            if (close[i] > camarilla_l4_aligned[i] and 
                above_ema and 
                volume_confirmed):
                position = 1
                signals[i] = position_size
            # Enter short: price breaks below Camarilla H4 + below 1d EMA200 + volume
            elif (close[i] < camarilla_h4_aligned[i] and 
                  below_ema and 
                  volume_confirmed):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to 1d EMA200 or breaks below Camarilla H4
            if close[i] < ema_1d_aligned[i] or close[i] < camarilla_h4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to 1d EMA200 or breaks above Camarilla L4
            if close[i] > ema_1d_aligned[i] or close[i] > camarilla_l4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_Camarilla_EMA200_Volume_v1"
timeframe = "4h"
leverage = 1.0