#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d EMA34 trend filter and volume spike
# Williams Alligator: Jaw (SMA13), Teeth (SMA8), Lips (SMA5)
# Lips crossing above Teeth and Jaw = bullish; crossing below = bearish
# Trend filter: 1d EMA34 to align with higher timeframe direction
# Volume spike: >2x 20-period average to confirm momentum
# Designed for 12h timeframe to target 12-37 trades/year per symbol.
# Works in both bull (captures trend) and bear (avoids false breaks via trend filter)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for trend filter (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # 1d EMA(34) for higher timeframe trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Williams Alligator components (using 12h data)
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().values  # SMA13
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().values   # SMA8
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().values    # SMA5
    
    # Alligator signals: Lips crossing Teeth/Jaw
    lips_above_teeth = lips > teeth
    lips_above_jaw = lips > jaw
    teeth_above_jaw = teeth > jaw
    
    # Bullish: Lips above Teeth and Jaw, and Teeth above Jaw
    bullish_setup = lips_above_teeth & lips_above_jaw & teeth_above_jaw
    # Bearish: Lips below Teeth and Jaw, and Teeth below Jaw
    bearish_setup = (lips < teeth) & (lips < jaw) & (teeth < jaw)
    
    # Volume spike filter (20-period on 12h data)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bullish setup + 1d uptrend + volume spike
            if (bullish_setup[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Bearish setup + 1d downtrend + volume spike
            elif (bearish_setup[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: Alligator lines cross or trend reversal
            if position == 1:
                # Exit on bearish setup or trend reversal
                if (bearish_setup[i] or 
                    close[i] < ema_34_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit on bullish setup or trend reversal
                if (bullish_setup[i] or 
                    close[i] > ema_34_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12h_WilliamsAlligator_1dEMA34_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0