#!/usr/bin/env python3
"""
4h_Williams_Alligator_Touch_EMA34_Volume
Hypothesis: Price touches the Alligator's Jaw (EMA13) during strong trends (EMA34) with volume confirmation.
Works in bull/bear by capturing pullbacks to the 13-period EMA within 34-period EMA trends.
Target: 20-30 trades/year to minimize fee drag while capturing high-probability trend continuations.
"""

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
    
    # Williams Alligator: Jaw (EMA13), Teeth (EMA8), Lips (EMA5)
    jaw = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    teeth = pd.Series(close).ewm(span=8, adjust=False, min_periods=8).mean().values
    lips = pd.Series(close).ewm(span=5, adjust=False, min_periods=5).mean().values
    
    # Trend filter: EMA34 on 12h
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Volume spike: >1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(35, 20)  # Warmup for EMA and indicators
    
    for i in range(start_idx, n):
        if (np.isnan(jaw[i]) or 
            np.isnan(teeth[i]) or
            np.isnan(lips[i]) or
            np.isnan(ema_34_12h_aligned[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        jaw_val = jaw[i]
        teeth_val = teeth[i]
        lips_val = lips[i]
        ema34 = ema_34_12h_aligned[i]
        vol_spike = volume_spike[i]
        
        # Alligator aligned: Jaw > Teeth > Lips = Uptrend, Jaw < Teeth < Lips = Downtrend
        uptrend = jaw_val > teeth_val and teeth_val > lips_val
        downtrend = jaw_val < teeth_val and teeth_val < lips_val
        
        if position == 0:
            # Long: Pullback to Jaw in uptrend with volume spike
            if uptrend and abs(price - jaw_val) < 0.001 * jaw_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: Pullback to Jaw in downtrend with volume spike
            elif downtrend and abs(price - jaw_val) < 0.001 * jaw_val and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: Price crosses Teeth (8) or trend changes
            if price < teeth_val or not uptrend:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: Price crosses Teeth (8) or trend changes
            if price > teeth_val or not downtrend:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Williams_Alligator_Touch_EMA34_Volume"
timeframe = "4h"
leverage = 1.0