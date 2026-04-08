#!/usr/bin/env python3
"""
1h Range Breakout with 4h Trend and 1d Volume Filter
Hypothesis: In ranging markets, breakouts from 1h consolidation zones (identified by low ATR)
continue in the direction of the 4h trend. Volume confirmation on 1d filters false breakouts.
Works in bull/bear by using 4h trend filter to avoid counter-trend breakouts.
Target: 15-30 trades/year on 1h timeframe (60-120 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_range_breakout_4h_trend_1d_volume_v1"
timeframe = "1h"
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
    
    # 4h trend: EMA(20) - EMA(50)
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema20_4h = pd.Series(close_4h).ewm(span=20, min_periods=20, adjust=False).mean().values
    ema50_4h = pd.Series(close_4h).ewm(span=50, min_periods=50, adjust=False).mean().values
    trend_4h = ema20_4h - ema50_4h
    trend_4h_aligned = align_htf_to_ltf(prices, df_4h, trend_4h)
    
    # 1d volume filter: volume > 1.5x 20-day average
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_filter_1d = volume_1d > (vol_ma_1d * 1.5)
    vol_filter_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_filter_1d)
    
    # 1h ATR(14) for consolidation detection
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.inf], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 1h consolidation: ATR < 0.5 * 20-period ATR mean
    atr_ma = pd.Series(atr).rolling(window=20, min_periods=20).mean().values
    consolidation = atr < (atr_ma * 0.5)
    
    # 1h range: 20-period high/low
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(trend_4h_aligned[i]) or np.isnan(vol_filter_1d_aligned[i]) or
            np.isnan(atr[i]) or np.isnan(atr_ma[i]) or
            np.isnan(high_20[i]) or np.isnan(low_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below 20-period low OR trend turns bearish
            if (close[i] <= low_20[i] or 
                trend_4h_aligned[i] < 0):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: price closes above 20-period high OR trend turns bullish
            if (close[i] >= high_20[i] or 
                trend_4h_aligned[i] > 0):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry
            # Long: breakout above 20-period high with bullish 4h trend and volume
            if (close[i] > high_20[i] and 
                trend_4h_aligned[i] > 0 and 
                consolidation[i] and
                vol_filter_1d_aligned[i]):
                position = 1
                signals[i] = 0.20
            # Short: breakdown below 20-period low with bearish 4h trend and volume
            elif (close[i] < low_20[i] and 
                  trend_4h_aligned[i] < 0 and 
                  consolidation[i] and
                  vol_filter_1d_aligned[i]):
                position = -1
                signals[i] = -0.20
    
    return signals