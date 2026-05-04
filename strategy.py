#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + 1d Trend + Volume Spike
# Williams Alligator (Jaw=TEETH=LIPS smoothed SMAs) identifies trend absence/presence.
# Long when price > LIPS and Jaw < TEETH < LIPS (bullish alignment) + 1d EMA50 uptrend + volume > 1.5x 20-period EMA volume.
# Short when price < LIPS and Jaw > TEETH > LIPS (bearish alignment) + 1d EMA50 downtrend + volume confirmation.
# Uses 12h timeframe targeting 50-150 total trades over 4 years (12-37/year).
# Discrete position sizing (0.25) to minimize fee churn and manage drawdown.

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
    
    # Get 12h data for Williams Alligator
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 13:
        return np.zeros(n)
    
    # Calculate 12h Williams Alligator: Jaw (SMA13), Teeth (SMA8), Lips (SMA5)
    close_12h = df_12h['close'].values
    jaw = pd.Series(close_12h).rolling(window=13, min_periods=13).mean().values
    teeth = pd.Series(close_12h).rolling(window=8, min_periods=8).mean().values
    lips = pd.Series(close_12h).rolling(window=5, min_periods=5).mean().values
    
    # Align Alligator lines to 12h timeframe (wait for completed 12h bar)
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips)
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
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
            # Long: price > Lips and Jaw < Teeth < Lips (bullish alignment) + 1d EMA50 uptrend + volume
            if (close[i] > lips_aligned[i] and 
                jaw_aligned[i] < teeth_aligned[i] < lips_aligned[i] and
                close[i] > ema_50_1d_aligned[i] and
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: price < Lips and Jaw > Teeth > Lips (bearish alignment) + 1d EMA50 downtrend + volume
            elif (close[i] < lips_aligned[i] and 
                  jaw_aligned[i] > teeth_aligned[i] > lips_aligned[i] and
                  close[i] < ema_50_1d_aligned[i] and
                  volume_confirm):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price < Lips OR Jaw > Teeth (trend weakness) OR 1d EMA50 turns down
            if (close[i] < lips_aligned[i] or 
                jaw_aligned[i] > teeth_aligned[i] or
                close[i] < ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price > Lips OR Jaw < Teeth (trend weakness) OR 1d EMA50 turns up
            if (close[i] > lips_aligned[i] or 
                jaw_aligned[i] < teeth_aligned[i] or
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals