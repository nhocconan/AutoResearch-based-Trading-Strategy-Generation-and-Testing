#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h Williams Alligator with 1d trend filter and volume confirmation
    # Alligator (SMAs: Jaw=13, Teeth=8, Lips=5) identifies trends via alignment
    # 1d EMA34 filters for higher timeframe trend direction
    # Volume spike (2x 20-period MA) confirms institutional participation
    # Works in bull/bear: Alligator alignment + volume confirms strong trends
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data for Alligator calculation
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Williams Alligator: SMAs of median price
    median_price_12h = (high_12h + low_12h) / 2 if False else (df_12h['high'].values + df_12h['low'].values) / 2
    jaw = pd.Series(median_price_12h).rolling(window=13, min_periods=13).mean().values  # Jaw (13)
    teeth = pd.Series(median_price_12h).rolling(window=8, min_periods=8).mean().values   # Teeth (8)
    lips = pd.Series(median_price_12h).rolling(window=5, min_periods=5).mean().values    # Lips (5)
    
    # Align Alligator lines to 12h timeframe (they're already 12h, but align for safety)
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips)
    
    # Load 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume spike filter (20-period)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma20  # Require 2x volume for confirmation
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Lips > Teeth > Jaw (bullish alignment) + above 1d EMA34 + volume spike
            if (lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i] and 
                close[i] > ema34_1d_aligned[i] and vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Lips < Teeth < Jaw (bearish alignment) + below 1d EMA34 + volume spike
            elif (lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i] and 
                  close[i] < ema34_1d_aligned[i] and vol_spike[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Alligator alignment breaks or reverse alignment
            if position == 1:
                # Exit long if alignment breaks down (lips <= teeth or teeth <= jaw)
                if not (lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short if alignment breaks up (lips >= teeth or teeth >= jaw)
                if not (lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12h_Williams_Alligator_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0