#!/usr/bin/env python3
"""
6h Camarilla Pivot + Weekly Trend + Volume Confirmation v1
Hypothesis: Camarilla pivot levels from daily data provide strong intraday support/resistance.
At 6h timeframe, we use weekly trend (1w EMA) to filter direction and daily volume to confirm breakouts.
Long when price breaks above R4 with weekly uptrend and volume spike.
Short when price breaks below S4 with weekly downtrend and volume spike.
This structure should work in both bull (breakouts continue) and bear (breakdowns continue) markets.
Target: 15-30 trades/year to avoid fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_camarilla_pivot_weekly_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate Camarilla levels from previous day
    # PP = (H + L + C) / 3
    # R4 = PP + ((H - L) * 1.1)
    # S4 = PP - ((H - L) * 1.1)
    pp = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    r4 = pp + ((df_1d['high'] - df_1d['low']) * 1.1)
    s4 = pp - ((df_1d['high'] - df_1d['low']) * 1.1)
    
    # Align to 6h timeframe (shifted by 1 day for prior day's levels)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4.values)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4.values)
    
    # Weekly EMA(34) for trend filter
    ema_34_1w = df_1w['close'].ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume filter (>1.8x 24-period average = ~3 days)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_filter = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(24, n):
        # Skip if any required data is NaN
        if (np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below S4 or weekly trend turns down
            if close[i] <= s4_aligned[i] or close[i] < ema_34_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above R4 or weekly trend turns up
            if close[i] >= r4_aligned[i] or close[i] > ema_34_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long breakout at R4 with weekly uptrend and volume
            if (close[i] >= r4_aligned[i] and 
                close[i] > ema_34_1w_aligned[i] and 
                vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short breakdown at S4 with weekly downtrend and volume
            elif (close[i] <= s4_aligned[i] and 
                  close[i] < ema_34_1w_aligned[i] and 
                  vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals