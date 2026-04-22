#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d trend filter and volume confirmation
# Williams Alligator uses three SMAs (Jaw: 13, Teeth: 8, Lips: 5) to identify trends.
# In an uptrend: Lips > Teeth > Jaw; in downtrend: Lips < Teeth < Jaw.
# We go long when Alligator is bullish (Lips > Teeth > Jaw) AND price > 1d EMA(34) AND volume > 2x average.
# We go short when Alligator is bearish (Lips < Teeth < Jaw) AND price < 1d EMA(34) AND volume > 2x average.
# This filters choppy markets and aligns with higher timeframe trend, reducing whipsaws.
# Target: 12-37 trades/year (50-150 total over 4 years) to avoid fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for EMA trend (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # 1d EMA(34) for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Williams Alligator components on 12h data
    # Jaw: 13-period SMMA (smoothed moving average)
    # Teeth: 8-period SMMA
    # Lips: 5-period SMMA
    # SMMA is similar to EMA but with different smoothing; we'll use EMA as proxy for simplicity
    jaw = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    teeth = pd.Series(close).ewm(span=8, adjust=False, min_periods=8).mean().values
    lips = pd.Series(close).ewm(span=5, adjust=False, min_periods=5).mean().values
    
    # Volume confirmation: 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(lips[i]) or 
            np.isnan(teeth[i]) or np.isnan(jaw[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Lips > Teeth > Jaw (bullish alignment) + price above 1d EMA + volume spike
            if (lips[i] > teeth[i] and teeth[i] > jaw[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                volume[i] > 2.0 * vol_avg_20[i]):
                signals[i] = 0.25
                position = 1
            # Short: Lips < Teeth < Jaw (bearish alignment) + price below 1d EMA + volume spike
            elif (lips[i] < teeth[i] and teeth[i] < jaw[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume[i] > 2.0 * vol_avg_20[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Alligator alignment changes or price crosses 1d EMA
            if position == 1:
                # Exit long: Alligator turns bearish OR price crosses below 1d EMA
                if (lips[i] <= teeth[i] or teeth[i] <= jaw[i] or 
                    close[i] < ema_34_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short: Alligator turns bullish OR price crosses above 1d EMA
                if (lips[i] >= teeth[i] or teeth[i] >= jaw[i] or 
                    close[i] > ema_34_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12h_WilliamsAlligator_1dEMA34_VolumeConfirmation"
timeframe = "12h"
leverage = 1.0