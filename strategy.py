#!/usr/bin/env python3
"""
12h_Camarilla_Pivot_Reversal_1dTrend_Volume
Hypothesis: On 12h timeframe, price reversals at daily Camarilla pivot levels (S1/R1) with
volume confirmation and 1d trend alignment capture mean-reversion bounces in ranging markets
and trend continuations in trending markets. This strategy avoids false breakouts by
requiring price to approach but not break the pivot level, then reverse with volume.
Works in both bull and bear markets by adapting to regime via trend filter.
Target: 15-25 trades/year per symbol to minimize fee drag.
"""

name = "12h_Camarilla_Pivot_Reversal_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels for each daily bar
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12, PP = (H+L+C)/3
    h_1d = df_1d['high'].values
    l_1d = df_1d['low'].values
    c_1d = df_1d['close'].values
    
    camarilla_pp = (h_1d + l_1d + c_1d) / 3.0
    camarilla_r1 = c_1d + (h_1d - l_1d) * 1.1 / 12.0
    camarilla_s1 = c_1d - (h_1d - l_1d) * 1.1 / 12.0
    
    # Align Camarilla levels to 12h chart (no additional delay needed for pivot levels)
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pp)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Get 1d trend filter: EMA34
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume confirmation: current volume > 1.5x 24-period average (12 days on 12h)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup
        if position == 0:
            # LONG: Price approaches S1 from below and reverses up with volume in uptrend
            if (low[i] <= camarilla_s1_aligned[i] * 1.001 and  # Allow 0.1% tolerance
                close[i] > camarilla_s1_aligned[i] and
                volume_filter[i] and
                close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price approaches R1 from above and reverses down with volume in downtrend
            elif (high[i] >= camarilla_r1_aligned[i] * 0.999 and  # Allow 0.1% tolerance
                  close[i] < camarilla_r1_aligned[i] and
                  volume_filter[i] and
                  close[i] < ema34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price reaches pivot point or trend reverses
            if (close[i] >= camarilla_pp_aligned[i]) or \
               (close[i] < ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price reaches pivot point or trend reverses
            if (close[i] <= camarilla_pp_aligned[i]) or \
               (close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals