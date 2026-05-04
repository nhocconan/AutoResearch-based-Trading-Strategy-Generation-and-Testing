#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + 1d EMA50 Trend + Volume Spike
# Williams Alligator (JAW=13, TEETH=8, LIPS=5) identifies trending vs ranging markets.
# Long when LIPS > TEETH > JAW (bullish alignment) + price > 1d EMA50 + volume > 1.5x 20-period EMA volume.
# Short when LIPS < TEETH < JAW (bearish alignment) + price < 1d EMA50 + volume confirmation.
# Designed for 12h timeframe targeting 50-150 total trades over 4 years (12-37/year).
# Uses discrete position sizing (0.25) to minimize fee churn and manage drawdown.

name = "12h_WilliamsAlligator_1dEMA50_Trend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_ = prices['open'].values
    
    # Get 1d data for Williams Alligator and EMA50 trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d Williams Alligator
    # Median price = (high + low) / 2
    median_price = (df_1d['high'] + df_1d['low']) / 2
    
    # JAW: 13-period SMMA, shifted 8 bars
    jaw = median_price.rolling(window=13, min_periods=13).mean().shift(8)
    # TEETH: 8-period SMMA, shifted 5 bars
    teeth = median_price.rolling(window=8, min_periods=8).mean().shift(5)
    # LIPS: 5-period SMMA, shifted 3 bars
    lips = median_price.rolling(window=5, min_periods=5).mean().shift(3)
    
    jaw_vals = jaw.values
    teeth_vals = teeth.values
    lips_vals = lips.values
    
    # Align Williams Alligator lines to 12h timeframe (wait for completed 1d bar)
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw_vals)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth_vals)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips_vals)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: 20-period EMA of volume on 12h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.5 x 20-period EMA
        volume_confirm = volume[i] > (1.5 * vol_ema_20[i])
        
        if position == 0:
            # Long: Lips > Teeth > Jaw (bullish alignment) + price > 1d EMA50 + volume
            if (lips_aligned[i] > teeth_aligned[i] and 
                teeth_aligned[i] > jaw_aligned[i] and
                close[i] > ema_50_1d_aligned[i] and
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: Lips < Teeth < Jaw (bearish alignment) + price < 1d EMA50 + volume
            elif (lips_aligned[i] < teeth_aligned[i] and 
                  teeth_aligned[i] < jaw_aligned[i] and
                  close[i] < ema_50_1d_aligned[i] and
                  volume_confirm):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: bullish alignment broken OR price < 1d EMA50
            if (lips_aligned[i] <= teeth_aligned[i] or 
                teeth_aligned[i] <= jaw_aligned[i] or
                close[i] < ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: bearish alignment broken OR price > 1d EMA50
            if (lips_aligned[i] >= teeth_aligned[i] or 
                teeth_aligned[i] >= jaw_aligned[i] or
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals