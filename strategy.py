#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike
Hypothesis: Trade 12h timeframe using Camarilla pivot (R1/S1) breakouts filtered by 1d EMA34 trend and volume spike.
Enter long when price breaks above R1 AND 1d trend is bullish (close > EMA34) AND volume > 1.5x 20-period average.
Enter short when price breaks below S1 AND 1d trend is bearish (close < EMA34) AND volume > 1.5x 20-period average.
Exit when price re-enters the Camarilla H3/L3 range or 1d trend reverses.
Uses discrete sizing 0.25 to manage risk. Target 12-37 trades/year on 12h timeframe.
Camarilla levels provide precise intraday support/resistance. Volume spike confirms institutional interest.
1d EMA34 filter ensures we only trade with the higher timeframe trend, avoiding counter-trend whipsaws in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from previous 1d bar (H1d, L1d, C1d)
    # We need to align the previous day's HLC to current 12h bars
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Typical price for Camarilla calculation
    typical_price_1d = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    R1 = close_1d + range_1d * 1.1 / 12
    S1 = close_1d - range_1d * 1.1 / 12
    R3 = close_1d + range_1d * 1.1 / 4
    S3 = close_1d - range_1d * 1.1 / 4
    # H3/L3 for exit (upper/lower bounds of trading range)
    H3 = close_1d + range_1d * 1.1 / 2
    L3 = close_1d - range_1d * 1.1 / 2
    
    # Align all levels to 12h timeframe (wait for 1d bar to close)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    
    # Volume spike: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for 1d EMA34 (34) and volume MA (20)
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or
            np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R1 AND 1d trend bullish (close > EMA34) AND volume spike
            long_setup = (close[i] > R1_aligned[i]) and \
                         (close[i] > ema_34_1d_aligned[i]) and \
                         volume_spike[i]
            # Short: price breaks below S1 AND 1d trend bearish (close < EMA34) AND volume spike
            short_setup = (close[i] < S1_aligned[i]) and \
                          (close[i] < ema_34_1d_aligned[i]) and \
                          volume_spike[i]
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price re-enters H3/L3 range (closes below H3) OR 1d trend turns bearish
            if (close[i] < H3_aligned[i]) or \
               (close[i] < ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price re-enters H3/L3 range (closes above L3) OR 1d trend turns bullish
            if (close[i] > L3_aligned[i]) or \
               (close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0