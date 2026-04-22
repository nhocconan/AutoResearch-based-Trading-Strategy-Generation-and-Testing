#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + 1w Trend Filter + Volume Spike
# Williams Alligator: Jaw (13-period SMA, 8 bars ahead), Teeth (8-period SMA, 5 bars ahead), Lips (5-period SMA, 3 bars ahead)
# Trend: Jaw > Teeth > Lips = uptrend, Jaw < Teeth < Lips = downtrend
# Entry: Alligator aligned in direction + price outside mouth + 1w trend confirmation + volume spike
# Exit: Alligator lines cross or trend reversal
# Works in trending markets (both bull/bear) by filtering with higher timeframe trend
# Target: 15-25 trades/year per symbol with strict confluence

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data for trend filter (ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # 1w EMA(34) for higher timeframe trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Williams Alligator components (using 6h data)
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().shift(5).values
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Alligator alignment
    jaw_above_teeth = jaw > teeth
    teeth_above_lips = teeth > lips
    jaw_below_teeth = jaw < teeth
    teeth_below_lips = teeth < lips
    
    # Mouth open conditions (avoid chop)
    jaw_lips_distance = np.abs(jaw - lips)
    jaw_lips_ma = pd.Series(jaw_lips_distance).rolling(window=20, min_periods=20).mean().values
    mouth_open = jaw_lips_distance > (jaw_lips_ma * 0.5)
    
    # Price relative to Alligator
    price_above_jaw = close > jaw
    price_below_jaw = close < jaw
    
    # Volume spike filter (20-period on 6h data)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Jaw > Teeth > Lips (uptrend) + price above Jaw + 1w uptrend + volume spike + mouth open
            if (jaw_above_teeth[i] and teeth_above_lips[i] and 
                price_above_jaw[i] and 
                close[i] > ema_34_1w_aligned[i] and 
                vol_spike[i] and mouth_open[i]):
                signals[i] = 0.25
                position = 1
            # Short: Jaw < Teeth < Lips (downtrend) + price below Jaw + 1w downtrend + volume spike + mouth open
            elif (jaw_below_teeth[i] and teeth_below_lips[i] and 
                  price_below_jaw[i] and 
                  close[i] < ema_34_1w_aligned[i] and 
                  vol_spike[i] and mouth_open[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: Alligator lines cross or trend reversal
            if position == 1:
                # Exit on Alligator turning down or trend reversal
                if (jaw[i] <= teeth[i] or 
                    teeth[i] <= lips[i] or 
                    close[i] < ema_34_1w_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit on Alligator turning up or trend reversal
                if (jaw[i] >= teeth[i] or 
                    teeth[i] >= lips[i] or 
                    close[i] > ema_34_1w_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_WilliamsAlligator_1wEMA34_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0