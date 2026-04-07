#!/usr/bin/env python3
"""
1h_atr_breakout_4h1d_trend_volume_v1
Hypothesis: On 1h timeframe, use ATR breakout with 4h trend filter (EMA) and 1d volume confirmation to capture strong trending moves. Enter long when price breaks above ATR-based upper band with 4h uptrend and high volume; enter short when price breaks below ATR-based lower band with 4h downtrend and high volume. Exit when price reverses back through the ATR-based middle band. Uses 4h for trend direction, 1d for volume regime filter, and 1h for precise entry timing. Designed to work in both bull and bear markets by requiring strong trend alignment and volume confirmation, reducing false signals and controlling trade frequency to 15-37/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_atr_breakout_4h1d_trend_volume_v1"
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
    
    # ATR calculation (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 4h EMA for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    ema_4h = pd.Series(df_4h['close'].values).ewm(span=20, min_periods=20).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # 1d volume average for regime filter (20-period)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    vol_20d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_20d_aligned = align_htf_to_ltf(prices, df_1d, vol_20d)
    
    # ATR-based bands (1.5 * ATR)
    upper_band = close + 1.5 * atr
    lower_band = close - 1.5 * atr
    middle_band = close  # using close as midpoint for exit
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(14, n):
        # Skip if required data not available
        if (np.isnan(atr[i]) or np.isnan(ema_4h_aligned[i]) or 
            np.isnan(vol_20d_aligned[i]) or vol_20d_aligned[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume condition: current volume > 1.5x 20-day average
        vol_filter = volume[i] > 1.5 * vol_20d_aligned[i]
        
        if position == 1:  # Long position
            # Exit when price crosses below middle band
            if close[i] < middle_band[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit when price crosses above middle band
            if close[i] > middle_band[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry
            # Long entry: price breaks above upper band + 4h uptrend + volume
            long_entry = (close[i] > upper_band[i] and 
                         close[i] > ema_4h_aligned[i] and 
                         vol_filter)
            
            # Short entry: price breaks below lower band + 4h downtrend + volume
            short_entry = (close[i] < lower_band[i] and 
                          close[i] < ema_4h_aligned[i] and 
                          vol_filter)
            
            if long_entry:
                position = 1
                signals[i] = 0.20
            elif short_entry:
                position = -1
                signals[i] = -0.20
    
    return signals