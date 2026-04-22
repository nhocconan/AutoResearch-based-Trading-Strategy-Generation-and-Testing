#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Williams Alligator with 1d EMA34 trend filter and volume spike
    # Williams Alligator (13,8,5 SMAs) identifies trend direction and strength
    # EMA34 on 1d filters for long-term trend (more stable than shorter TF)
    # Volume spike (2x 20-period MA) confirms institutional participation
    # Works in bull/bear: Alligator convergence/divergence + volume confirms trend
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Williams Alligator on 4h data (13,8,5 SMAs)
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().values  # 13-period SMA
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().values   # 8-period SMA
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().values    # 5-period SMA
    
    # Volume spike filter (20-period)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma20  # Require 2x volume for confirmation
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(jaw[i]) or 
            np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or 
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Lips > Teeth > Jaw (bullish alignment) + volume spike + price above EMA34
            if lips[i] > teeth[i] and teeth[i] > jaw[i] and vol_spike[i] and close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Lips < Teeth < Jaw (bearish alignment) + volume spike + price below EMA34
            elif lips[i] < teeth[i] and teeth[i] < jaw[i] and vol_spike[i] and close[i] < ema34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Alligator lines cross (trend weakening)
            if position == 1:
                if lips[i] < teeth[i]:  # Bullish alignment broken
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if lips[i] > teeth[i]:  # Bearish alignment broken
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_WilliamsAlligator_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0