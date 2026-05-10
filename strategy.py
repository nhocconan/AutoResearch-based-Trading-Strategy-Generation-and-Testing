#!/usr/bin/env python3
# 4h_Williams_Alligator_1dTrend_VolumeFilter
# Hypothesis: Williams Alligator (Jaw=13, Teeth=8, Lips=5) with 1d EMA trend filter and volume spike confirmation.
# Alligator identifies trend strength: converging lines = ranging (avoid), diverging lines = trending (trade).
# Long when Lips > Teeth > Jaw in uptrend with volume spike; Short when Lips < Teeth < Jaw in downtrend with volume spike.
# Works in bull/bear by aligning with daily trend direction. Targets 20-50 trades/year to minimize fee drag.

name = "4h_Williams_Alligator_1dTrend_VolumeFilter"
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
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Williams Alligator: SMAs with future shift (Jaw=13, Teeth=8, Lips=5)
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().shift(5).values
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Get daily EMA for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation (20-period MA)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(13, 8, 5, 34, 20) + 8  # Warmup for Alligator + shifts + daily EMA + volume MA
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Daily trend filter
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        
        # Volume confirmation
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        # Alligator alignment: Lips > Teeth > Jaw = bullish alignment
        bullish_alignment = lips[i] > teeth[i] and teeth[i] > jaw[i]
        # Alligator alignment: Lips < Teeth < Jaw = bearish alignment
        bearish_alignment = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        if position == 0:
            # Long entry: bullish Alligator alignment + uptrend + volume spike
            if bullish_alignment and uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: bearish Alligator alignment + downtrend + volume spike
            elif bearish_alignment and downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Alligator convergence (Lips <= Teeth) or trend reversal
            if lips[i] <= teeth[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Alligator convergence (Lips >= Teeth) or trend reversal
            if lips[i] >= teeth[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals