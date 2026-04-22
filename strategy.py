# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    """
    Hypothesis: 12h Williams Alligator with 1w EMA50 trend filter and volume confirmation
    Williams Alligator (Jaw/Teeth/Lips) identifies trend direction and strength
    1w EMA50 filters for long-term trend direction to avoid counter-trend trades
    Volume spike confirms institutional participation
    Works in bull/bear: trades in direction of long-term trend with Alligator alignment
    """
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data for Williams Alligator calculation
    df_12h = get_htf_data(prices, '12h')
    median_price_12h = (df_12h['high'] + df_12h['low']) / 2
    
    # Williams Alligator: three SMAs (Jaw=13, Teeth=8, Lips=5) shifted forward
    jaw = median_price_12h.rolling(window=13, min_periods=13).mean().shift(8).values
    teeth = median_price_12h.rolling(window=8, min_periods=8).mean().shift(5).values
    lips = median_price_12h.rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Align Alligator lines to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips)
    
    # Load 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    ema50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume spike filter (20-period)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma20  # Require 2x volume for confirmation
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Lips > Teeth > Jaw (bullish alignment) AND price > EMA50 (uptrend) AND volume spike
            if lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i] and \
               close[i] > ema50_1w_aligned[i] and vol_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Lips < Teeth < Jaw (bearish alignment) AND price < EMA50 (downtrend) AND volume spike
            elif lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i] and \
                 close[i] < ema50_1w_aligned[i] and vol_spike[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Alligator lines cross (trend weakening)
            if position == 1:
                if lips_aligned[i] < teeth_aligned[i]:  # Bullish alignment broken
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if lips_aligned[i] > teeth_aligned[i]:  # Bearish alignment broken
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12h_Williams_Alligator_1wEMA50_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0