#!/usr/bin/env python3
# Hypothesis: 4h Camarilla R1/S1 breakout with 1d EMA34 trend filter and volume confirmation (>1.8x 30-bar avg volume).
# Uses tighter volume threshold and discrete position sizing (0.25) to reduce overtrading.
# Designed to work in both bull and bear markets by combining price structure (Camarilla), trend (EMA34), and participation (volume spike).
# Target: 75-150 total trades over 4 years on 4h timeframe.

name = "4h_Camarilla_R1S1_Breakout_1dEMA34_Volume_Tight_v3"
timeframe = "4h"
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
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d Camarilla levels (based on prior 1d bar)
    prior_1d_high = df_1d['high'].values
    prior_1d_low = df_1d['low'].values
    prior_1d_close = df_1d['close'].values
    
    camarilla_r1 = prior_1d_close + 1.1 * (prior_1d_high - prior_1d_low) * 1.05 / 4
    camarilla_s1 = prior_1d_close - 1.1 * (prior_1d_high - prior_1d_low) * 1.05 / 4
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Calculate average volume for confirmation (30-period)
    lookback_vol = 30
    avg_volume = pd.Series(volume).rolling(window=lookback_vol, min_periods=lookback_vol).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(max(lookback_vol, 1), n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Camarilla R1, close > 1d EMA34, volume spike (>1.8x)
            if (high[i] > camarilla_r1_aligned[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                volume[i] > 1.8 * avg_volume[i]):
                signals[i] = 0.25  # Reduced position size to minimize churn
                position = 1
            # SHORT: Price breaks below Camarilla S1, close < 1d EMA34, volume spike (>1.8x)
            elif (low[i] < camarilla_s1_aligned[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume[i] > 1.8 * avg_volume[i]):
                signals[i] = -0.25  # Reduced position size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # CONTINUE LONG: Maintain full position if still above R1 and volume OK
            if (high[i] > camarilla_r1_aligned[i] and 
                volume[i] > avg_volume[i]):
                signals[i] = 0.25  # Maintain full position
            else:
                signals[i] = 0.0  # Exit if breaks below R1 or low volume
                position = 0
        elif position == -1:
            # CONTINUE SHORT: Maintain full position if still below S1 and volume OK
            if (low[i] < camarilla_s1_aligned[i] and 
                volume[i] > avg_volume[i]):
                signals[i] = -0.25  # Maintain full position
            else:
                signals[i] = 0.0  # Exit if breaks above S1 or low volume
                position = 0
    
    return signals