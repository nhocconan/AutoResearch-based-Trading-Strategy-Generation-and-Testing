#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla R1/S1 Breakout with 4h EMA34 Trend Filter and Volume Spike
- Camarilla R1/S1 levels from 4h act as intraday support/resistance; breakout with volume indicates strong continuation
- 4h EMA34 defines the medium-term trend: only long when price > EMA34, short when price < EMA34
- Volume confirmation (> 1.5x 20-period MA) reduces false breakouts
- Session filter (08-20 UTC) avoids low-liquidity hours
- Designed for 1h timeframe with tight entry conditions to target 60-150 total trades over 4 years
- Uses inner Camarilla levels (R1/S1) for precise entries and 4h trend filter for alignment
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 4h Camarilla pivots (based on previous 4h bar)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Camarilla levels: R1/S1 = C ± (H-L)*1.1/12 (inner levels)
    camarilla_r1 = close_4h + (high_4h - low_4h) * 1.1 / 12
    camarilla_s1 = close_4h - (high_4h - low_4h) * 1.1 / 12
    
    # Align to 1h timeframe (use previous 4h bar's levels for breakout)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s1)
    
    # Calculate 4h EMA34 for trend filter
    ema_34_4h = pd.Series(close_4h).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC (avoid low-liquidity hours)
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(1, 34, 20)  # need 4h pivots, 4h EMA34, vol MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(ema_34_4h_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: only trade between 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if position == 0 and in_session:
            # Long: price breaks above R1 AND above 4h EMA34 AND volume spike
            if (close[i] > camarilla_r1_aligned[i] and 
                close[i] > ema_34_4h_aligned[i] and 
                volume[i] > 1.5 * vol_ma[i]):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below S1 AND below 4h EMA34 AND volume spike
            elif (close[i] < camarilla_s1_aligned[i] and 
                  close[i] < ema_34_4h_aligned[i] and 
                  volume[i] > 1.5 * vol_ma[i]):
                signals[i] = -0.20
                position = -1
        else:
            # Exit: price returns to Camarilla H3/L3 levels OR crosses 4h EMA34
            exit_signal = False
            # Calculate H3/L3 for exit (Camarilla levels)
            camarilla_h3 = close_4h + (high_4h - low_4h) * 1.1 / 6
            camarilla_l3 = close_4h - (high_4h - low_4h) * 1.1 / 6
            camarilla_h3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_h3)
            camarilla_l3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_l3)
            
            if position == 1:
                # Exit long when price > H3 OR < 4h EMA34
                if close[i] > camarilla_h3_aligned[i] or close[i] < ema_34_4h_aligned[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short when price < L3 OR > 4h EMA34
                if close[i] < camarilla_l3_aligned[i] or close[i] > ema_34_4h_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1h_Camarilla_R1S1_Breakout_4hEMA34_Trend_VolumeSpike"
timeframe = "1h"
leverage = 1.0