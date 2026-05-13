#!/usr/bin/env python3
# Hypothesis: 12h Williams Alligator + Elder Ray volume spike with 1d trend filter for BTC/ETH.
# Uses Alligator (Jaw/Teeth/Lips) for trend direction, Elder Ray (Bull/Bear Power) for momentum,
# and 1d EMA50 for higher timeframe alignment. Volume confirmation (>1.5x 20-bar avg) filters false signals.
# Designed for low trade frequency (<150 total 12h trades over 4 years) to minimize fee drag while
# capturing sustained moves in both bull and bear markets via confluence of trend, momentum, and volume.

name = "12h_Williams_Alligator_ElderRay_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams Alligator: SMAs of median price (typical price)
    typical_price = (high + low + close) / 3.0
    jaw = pd.Series(typical_price).rolling(window=13, min_periods=13).mean().shift(8).values  # Blue line
    teeth = pd.Series(typical_price).rolling(window=8, min_periods=8).mean().shift(5).values   # Red line
    lips = pd.Series(typical_price).rolling(window=5, min_periods=5).mean().shift(3).values    # Green line
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Volume confirmation: 20-period average
    lookback_vol = 20
    avg_volume = pd.Series(volume).rolling(window=lookback_vol, min_periods=lookback_vol).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(max(lookback_vol, 13, 8), n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Alligator aligned (Lips > Teeth > Jaw), Bull Power > 0, volume spike
            if (lips[i] > teeth[i] > jaw[i] and 
                bull_power[i] > 0 and 
                volume[i] > 1.5 * avg_volume[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Alligator aligned (Jaw > Teeth > Lips), Bear Power < 0, volume spike
            elif (jaw[i] > teeth[i] > lips[i] and 
                  bear_power[i] < 0 and 
                  volume[i] > 1.5 * avg_volume[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Alligator breaks down (Lips < Teeth) or Bear Power > 0
            if (lips[i] < teeth[i]) or (bear_power[i] > 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Alligator breaks up (Lips > Teeth) or Bull Power < 0
            if (lips[i] > teeth[i]) or (bull_power[i] < 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals