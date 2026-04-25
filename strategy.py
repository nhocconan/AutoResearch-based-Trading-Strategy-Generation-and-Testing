#!/usr/bin/env python3
"""
6h Williams Alligator + 1d EMA34 trend + volume spike
Hypothesis: Williams Alligator (JAW/TEETH/LIPS) identifies trend phases on 6h. Only trade in alignment with 1d EMA34 trend filter. Volume confirmation ensures breakout strength. Discrete sizing (0.25) manages drawdown. Works in bull/bear via trend filter. Targets 50-150 trades over 4 years on 6h.
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
    
    # Load 1d data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Williams Alligator on 6h: SMMA(13,8), SMMA(8,5), SMMA(5,3)
    # SMMA = smoothed moving average (EMA with alpha=1/period)
    def smma(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan, dtype=float)
        result = np.full_like(arr, np.nan, dtype=float)
        # First value = SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CLOSE) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(high, 13)  # Alligator Jaw (blue) - SMMA of Median Price, 13 periods, 8 ahead
    teeth = smma(high, 8)  # Alligator Teeth (red) - SMMA of Median Price, 8 periods, 5 ahead
    lips = smma(high, 5)   # Alligator Lips (green) - SMMA of Median Price, 5 periods, 3 ahead
    
    # Align Alligator lines
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)  # Using 1d as reference for alignment (will be replaced)
    # Actually need to compute Alligator on 6h data directly
    # Let's recompute properly
    
    # Recompute Alligator on actual 6h prices
    median_price = (high + low) / 2
    jaw = smma(median_price, 13)
    teeth = smma(median_price, 8)
    lips = smma(median_price, 5)
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for all indicators
    start_idx = max(34, 13, 20) + 5
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        vol_spike = volume_spike[i]
        
        # Trend filter: price relative to 1d EMA34
        bullish_bias = curr_close > ema_1d_aligned[i]
        bearish_bias = curr_close < ema_1d_aligned[i]
        
        # Alligator alignment: Lips > Teeth > Jaw = bullish, Lips < Teeth < Jaw = bearish
        bullish_alligator = lips[i] > teeth[i] and teeth[i] > jaw[i]
        bearish_alligator = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        if position == 0:
            # Look for entry signals - require: Alligator alignment + trend + volume spike
            # Long: bullish Alligator AND bullish bias AND volume spike
            long_entry = bullish_alligator and bullish_bias and vol_spike
            # Short: bearish Alligator AND bearish bias AND volume spike
            short_entry = bearish_alligator and bearish_bias and vol_spike
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: bearish Alligator alignment OR loss of bullish bias
            if (not bullish_alligator) or (curr_close < ema_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: bullish Alligator alignment OR loss of bearish bias
            if (not bearish_alligator) or (curr_close > ema_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsAlligator_1dEMA34_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0