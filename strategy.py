#!/usr/bin/env python3
# Hypothesis: 6h Williams Alligator + Elder Ray (Bull/Bear Power) with 1d EMA34 trend filter and volume confirmation (>1.5x 20-bar avg volume).
# Uses Williams Alligator (JAW=TEETH=LIPS) to identify trend absence (all lines intertwined) vs presence (diverged lines).
# Elder Ray measures bull/bear power relative to 13-period EMA. Long when Bull Power > 0 and Bear Power < 0 (strong bullish momentum).
# Short when Bear Power > 0 and Bull Power < 0 (strong bearish momentum).
# 1d EMA34 ensures we only trade in alignment with higher timeframe trend.
# Volume confirmation filters low-momentum breakouts.
# Designed for low trade frequency (target: 50-150 total trades over 4 years) to minimize fee drag while capturing strong momentum moves.

name = "6h_WilliamsAlligator_ElderRay_1dEMA34_VolumeConfirm_v1"
timeframe = "6h"
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
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Williams Alligator: SMAs of median price (typical price)
    typical_price = (high + low + close) / 3.0
    # JAW: 13-period SMMA, 8 bars ahead
    jaw = pd.Series(typical_price).rolling(window=13, min_periods=13).mean().shift(8).values
    # TEETH: 8-period SMMA, 5 bars ahead
    teeth = pd.Series(typical_price).rolling(window=8, min_periods=8).mean().shift(5).values
    # LIPS: 5-period SMMA, 3 bars ahead
    lips = pd.Series(typical_price).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Alligator condition: lines are NOT intertwined (trending market)
    # JAW > TEETH > LIPS = uptrend, JAW < TEETH < LIPS = downtrend
    alligator_up = (jaw > teeth) & (teeth > lips)
    alligator_down = (jaw < teeth) & (teeth < lips)
    
    # Calculate average volume for confirmation (20-period)
    lookback_vol = 20
    avg_volume = pd.Series(volume).rolling(window=lookback_vol, min_periods=lookback_vol).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(max(lookback_vol, 13, 8), n):  # warmup for all indicators
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or 
            np.isnan(jaw[i]) or 
            np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Alligator uptrend, Bull Power > 0, Bear Power < 0, close > 1d EMA34, volume spike
            if (alligator_up[i] and 
                bull_power[i] > 0 and 
                bear_power[i] < 0 and 
                close[i] > ema_34_1d_aligned[i] and 
                volume[i] > 1.5 * avg_volume[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Alligator downtrend, Bear Power > 0, Bull Power < 0, close < 1d EMA34, volume spike
            elif (alligator_down[i] and 
                  bear_power[i] > 0 and 
                  bull_power[i] < 0 and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume[i] > 1.5 * avg_volume[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Alligator turns down OR Elder Ray turns bearish OR volume drops
            if (not alligator_up[i]) or (bull_power[i] <= 0) or (bear_power[i] >= 0) or (volume[i] < 0.6 * avg_volume[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Alligator turns up OR Elder Ray turns bullish OR volume drops
            if (not alligator_down[i]) or (bear_power[i] <= 0) or (bull_power[i] >= 0) or (volume[i] < 0.6 * avg_volume[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals