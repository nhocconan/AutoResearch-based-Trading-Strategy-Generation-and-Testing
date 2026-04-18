#!/usr/bin/env python3
"""
4h_KAMA_R1S1_Breakout_VolumeSpike_12hTrend
Hypothesis: KAMA trend direction + 12h Camarilla (R1/S1) breakouts with volume confirmation.
KAMA adapts to market noise, reducing whipsaw in ranging markets. Combined with 12h trend filter
and volume spike, this should yield high-quality trades in both bull and bear markets.
Target: 20-40 trades/year on 4h to minimize fee drag.
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
    
    # Get 12h data once
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h Camarilla pivot levels (R1, S1)
    high_12h = df_12h['high']
    low_12h = df_12h['low']
    close_12h = df_12h['close']
    
    pivot = (high_12h + low_12h + close_12h) / 3
    r1 = 2 * pivot - low_12h
    s1 = 2 * pivot - high_12h
    
    # Use previous 12h bar's levels to avoid look-ahead
    r1_prev = r1.shift(1).values
    s1_prev = s1.shift(1).values
    
    # Align to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_12h, r1_prev)
    s1_aligned = align_htf_to_ltf(prices, df_12h, s1_prev)
    
    # KAMA trend on 4h price
    def kama(close_series, er_length=10, fast_ema=2, slow_ema=30):
        change = abs(close_series - close_series.shift(er_length))
        volatility = abs(close_series.diff()).rolling(window=er_length).sum()
        er = change / volatility.replace(0, np.nan)
        sc = (er * (2/(fast_ema+1) - 2/(slow_ema+1)) + 2/(slow_ema+1)) ** 2
        kama_vals = [np.nan] * len(close_series)
        for i in range(1, len(close_series)):
            if np.isnan(sc.iloc[i]) or np.isnan(kama_vals[i-1]):
                kama_vals[i] = close_series.iloc[i]
            else:
                kama_vals[i] = kama_vals[i-1] + sc.iloc[i] * (close_series.iloc[i] - kama_vals[i-1])
        return np.array(kama_vals)
    
    kama_vals = kama(pd.Series(close))
    kama_dir = np.where(kama_vals > np.roll(kama_vals, 1), 1, -1)
    kama_dir[0] = 1  # initialize
    
    # Volume spike: 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or
            np.isnan(kama_vals[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        kama_direction = kama_dir[i]
        
        if position == 0:
            # Long: price breaks above R1 with volume spike and KAMA up
            if price > r1_val and volume_spike[i] and kama_direction > 0:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume spike and KAMA down
            elif price < s1_val and volume_spike[i] and kama_direction < 0:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long position
            signals[i] = 0.25
            # Exit: price returns to S1 or KAMA turns down
            if price <= s1_val or kama_direction < 0:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short position
            signals[i] = -0.25
            # Exit: price returns to R1 or KAMA turns up
            if price >= r1_val or kama_direction > 0:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_KAMA_R1S1_Breakout_VolumeSpike_12hTrend"
timeframe = "4h"
leverage = 1.0