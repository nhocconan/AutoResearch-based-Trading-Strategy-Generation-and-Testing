#!/usr/bin/env python3
"""
6h_camarilla_pivot_1d_ema_volume_v1
Hypothesis: On 6-hour timeframe, use daily Camarilla pivot levels with EMA filter and volume confirmation.
Go long when price breaks above R4 with volume > 1.5x average and EMA(50) > EMA(200).
Go short when price breaks below S4 with volume > 1.5x average and EMA(50) < EMA(200).
Exit when price returns to the daily pivot (PP) or EMA crossover reverses.
Designed for 15-30 trades/year to minimize fee dust while capturing strong breakouts in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_camarilla_pivot_1d_ema_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots and EMA
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # Calculate EMA on 1d close
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False).mean().values
    
    # Calculate Camarilla pivot levels for each day
    # Based on previous day's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point (PP) = (High + Low + Close) / 3
    pp = (high_1d + low_1d + close_1d) / 3.0
    
    # Camarilla levels
    # R4 = Close + ((High - Low) * 1.1/2)
    # R3 = Close + ((High - Low) * 1.1/4)
    # R2 = Close + ((High - Low) * 1.1/6)
    # R1 = Close + ((High - Low) * 1.1/12)
    # S1 = Close - ((High - Low) * 1.1/12)
    # S2 = Close - ((High - Low) * 1.1/6)
    # S3 = Close - ((High - Low) * 1.1/4)
    # S4 = Close - ((High - Low) * 1.1/2)
    r4 = close_1d + ((high_1d - low_1d) * 1.1 / 2)
    r3 = close_1d + ((high_1d - low_1d) * 1.1 / 4)
    s3 = close_1d - ((high_1d - low_1d) * 1.1 / 4)
    s4 = close_1d - ((high_1d - low_1d) * 1.1 / 2)
    
    # Align 1d data to 6h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Calculate average volume (50-period)
    vol_avg = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    
    # Session filter: 8-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after EMA warmup
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if data not available
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(ema_200_1d_aligned[i]) or
            np.isnan(pp_aligned[i]) or np.isnan(r4_aligned[i]) or
            np.isnan(s4_aligned[i]) or np.isnan(vol_avg[i])):
            signals[i] = 0.0
            continue
            
        # Volume filter: current volume > 1.5x average
        vol_filter = volume[i] > (1.5 * vol_avg[i])
        
        # EMA trend filter
        ema_bullish = ema_50_1d_aligned[i] > ema_200_1d_aligned[i]
        ema_bearish = ema_50_1d_aligned[i] < ema_200_1d_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price returns to pivot or EMA crossover reverses
            if close[i] <= pp_aligned[i] or not ema_bullish:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price returns to pivot or EMA crossover reverses
            if close[i] >= pp_aligned[i] or not ema_bearish:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Only enter with volume and EMA alignment
            if vol_filter:
                if ema_bullish and close[i] > r4_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                elif ema_bearish and close[i] < s4_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals