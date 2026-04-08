#!/usr/bin/env python3
"""
12h Camarilla Pivot Reversal with Volume Spike and 1d Trend Filter
Hypothesis: Price reversals at 1d Camarilla pivot levels (L3/H3) with volume spikes
capture mean-reversion in ranging markets while trend filter (1d EMA60) avoids
counter-trend trades in strong trends. Works in both bull/bear by taking longs
near support in uptrends and shorts near resistance in downtrends. Target: 15-30 trades/year on 12h.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_reversal_1d_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for Camarilla pivot and trend
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla pivot levels: H3, L3
    # H3 = High + 1.1*(High-Low)/6, L3 = Low - 1.1*(High-Low)/6
    range_1d = high_1d - low_1d
    camarilla_h3 = high_1d + 1.1 * range_1d / 6
    camarilla_l3 = low_1d - 1.1 * range_1d / 6
    
    # Align H3/L3 to 12h timeframe (shifted by 1 for completed bar)
    h3_12h = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_12h = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # 1d EMA(60) for trend filter
    ema_60_1d = pd.Series(close_1d).ewm(span=60, min_periods=60, adjust=False).mean().values
    ema_60_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_60_1d)
    
    # Volume spike: current volume > 2.0 x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):
        # Skip if any required data is NaN
        if (np.isnan(h3_12h[i]) or 
            np.isnan(l3_12h[i]) or 
            np.isnan(ema_60_1d_aligned[i]) or 
            np.isnan(vol_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price reaches H3 (resistance) OR trend turns bearish
            if (close[i] >= h3_12h[i] or 
                close[i] < ema_60_1d_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price reaches L3 (support) OR trend turns bullish
            if (close[i] <= l3_12h[i] or 
                close[i] > ema_60_1d_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry at reversal levels
            # Long near L3 support with volume spike and uptrend bias
            if (close[i] <= l3_12h[i] * 1.005 and  # within 0.5% of L3
                close[i] > ema_60_1d_aligned[i] and  # above trend (uptrend bias)
                vol_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short near H3 resistance with volume spike and downtrend bias
            elif (close[i] >= h3_12h[i] * 0.995 and  # within 0.5% of H3
                  close[i] < ema_60_1d_aligned[i] and  # below trend (downtrend bias)
                  vol_spike[i]):
                position = -1
                signals[i] = -0.25
    
    return signals