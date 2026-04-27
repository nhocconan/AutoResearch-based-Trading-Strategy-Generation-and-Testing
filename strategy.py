#/usr/bin/env python3
"""
12h_Camarilla_Pivot_R4_S4_Breakout_1dTrend_Volume
Hypothesis: Camarilla R4/S4 levels from daily pivot act as strong support/resistance. 
Breakouts with volume confirmation and daily EMA trend filter work in both bull/bear markets.
Target: 15-25 trades/year on 12h to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_htf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivot and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    # (H-L)/12 * multiplier + close
    range_1d = df_1d['high'] - df_1d['low']
    close_1d = df_1d['close']
    
    R4 = close_1d + (range_1d * 1.1 / 2)  # R4 = C + ((H-L) * 1.1/2)
    S4 = close_1d - (range_1d * 1.1 / 2)  # S4 = C - ((H-L) * 1.1/2)
    
    R4_vals = R4.values
    S4_vals = S4.values
    
    # Align to 12h timeframe
    R4_12h = align_ltf_to_htf(prices, df_1d, R4_vals)
    S4_12h = align_ltf_to_htf(prices, df_1d, S4_vals)
    
    # Daily trend: EMA34
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_12h = align_ltf_to_htf(prices, df_1d, ema34_1d)
    
    # Volume confirmation: volume > 2 * 24-period average (2 days of 12h bars)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for EMA and volume MA
    start_idx = max(34, 24)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if np.isnan(R4_12h[i]) or np.isnan(S4_12h[i]) or np.isnan(ema34_12h[i]):
            signals[i] = 0.0
            continue
        
        r4 = R4_12h[i]
        s4 = S4_12h[i]
        ema_trend = ema34_12h[i]
        vol_spike_val = vol_spike[i]
        
        if position == 0:
            # Long: break above R4 with volume + uptrend
            if close[i] > r4 and vol_spike_val and close[i] > ema_trend:
                signals[i] = size
                position = 1
            # Short: break below S4 with volume + downtrend
            elif close[i] < s4 and vol_spike_val and close[i] < ema_trend:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below S4 or trend reverses
            if close[i] < s4 or close[i] < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price breaks above R4 or trend reverses
            if close[i] > r4 or close[i] > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Camarilla_Pivot_R4_S4_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0