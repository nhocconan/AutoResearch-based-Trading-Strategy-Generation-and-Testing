#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + 1d EMA50 trend + volume confirmation
# Targets 12-37 trades/year (50-150 total over 4 years) to minimize fee drag
# Williams Alligator (Jaw/Teeth/Lips) identifies trend: Lips > Teeth > Jaw = bullish, inverse = bearish
# 1d EMA50 provides higher-timeframe trend bias: long when price > EMA50, short when price < EMA50
# Volume spike (1.5x 20-period average) confirms institutional participation
# Works in bull markets via trend-following Alligator alignment and bear markets via counter-trend fade
# Discrete position sizing: 0.25 (25% of capital) manages drawdown while capturing moves

name = "12h_WilliamsAlligator_1dEMA50_VolumeConfirm"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d Williams Alligator (smoothed medians)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    # Median price = (high + low + close) / 3
    median_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3.0
    median_price_vals = median_price.values
    
    # Alligator lines: Jaw (13-period, 8-shift), Teeth (8-period, 5-shift), Lips (5-period, 3-shift)
    jaw = pd.Series(median_price_vals).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth = pd.Series(median_price_vals).rolling(window=8, min_periods=8).mean().shift(5).values
    lips = pd.Series(median_price_vals).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Align to 12h timeframe (wait for completed 1d bar)
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Calculate 1d EMA50 trend (prior completed 1d bar's EMA)
    ema_50 = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Calculate 12h volume spike (1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(ema_50_aligned[i]) or 
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Alligator bullish alignment (Lips > Teeth > Jaw) AND price > 1d EMA50 AND volume spike
            if (lips_aligned[i] > teeth_aligned[i] and 
                teeth_aligned[i] > jaw_aligned[i] and 
                close[i] > ema_50_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: Alligator bearish alignment (Jaw > Teeth > Lips) AND price < 1d EMA50 AND volume spike
            elif (jaw_aligned[i] > teeth_aligned[i] and 
                  teeth_aligned[i] > lips_aligned[i] and 
                  close[i] < ema_50_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Alligator turns bearish (Jaw > Teeth) OR price falls below 1d EMA50
            if jaw_aligned[i] > teeth_aligned[i] or close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Alligator turns bullish (Teeth > Lips) OR price rises above 1d EMA50
            if teeth_aligned[i] > lips_aligned[i] or close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals