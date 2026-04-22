#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike
    # Camarilla levels provide institutional-grade support/resistance derived from prior day
    # EMA34 on 1d filters for medium-term trend direction
    # Volume spike (2x 20-period MA) confirms institutional participation
    # Works in bull/bear: breaks through key levels with trend and volume confirmation
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1h data for EMA34 trend (using 1h as proxy for trend, more responsive than 1d)
    df_1h = get_htf_data(prices, '1h')
    ema34_1h = pd.Series(df_1h['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1h_aligned = align_htf_to_ltf(prices, df_1h, ema34_1h)
    
    # Load 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    # Calculate Camarilla levels from previous 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True range for Camarilla calculation
    tr_1d = np.maximum(high_1d - low_1d, 
                       np.maximum(np.abs(high_1d - np.roll(close_1d, 1)),
                                  np.abs(low_1d - np.roll(close_1d, 1))))
    tr_1d[0] = high_1d[0] - low_1d[0]
    
    # Camarilla multiplier
    # R3/S3 are the most significant levels for breakouts
    camarilla_multiplier = 1.1 / 12
    
    # Calculate Camarilla levels for each 1d bar (based on previous day)
    camarilla_r3 = close_1d + (high_1d - low_1d) * camarilla_multiplier * 3
    camarilla_s3 = close_1d - (high_1d - low_1d) * camarilla_multiplier * 3
    
    # Align Camarilla levels to 4h timeframe (using previous day's levels)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume spike filter (20-period)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma20  # Require 2x volume for confirmation
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema34_1h_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above R3 with volume spike and price above 1h EMA34 (uptrend)
            if close[i] > camarilla_r3_aligned[i] and vol_spike[i] and close[i] > ema34_1h_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Break below S3 with volume spike and price below 1h EMA34 (downtrend)
            elif close[i] < camarilla_s3_aligned[i] and vol_spike[i] and close[i] < ema34_1h_aligned[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Return to opposite Camarilla level (S3 for longs, R3 for shorts)
            if position == 1:
                if close[i] < camarilla_s3_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > camarilla_r3_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R3_S3_Breakout_1hEMA34_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0