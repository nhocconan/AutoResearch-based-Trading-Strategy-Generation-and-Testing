#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Bollinger Bands breakout with 1d EMA34 trend filter and volume spike
    # Bollinger Bands provide volatility-based support/resistance
    # EMA34 on 1d filters for medium-term trend direction
    # Volume spike (2x 20-period MA) confirms institutional participation
    # Works in bull/bear: breaks through volatility bands with trend and volume confirmation
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Bollinger Bands (20, 2)
    bb_length = 20
    bb_mult = 2.0
    bb_basis = pd.Series(close).rolling(window=bb_length, min_periods=bb_length).mean().values
    bb_dev = bb_mult * pd.Series(close).rolling(window=bb_length, min_periods=bb_length).std().values
    bb_upper = bb_basis + bb_dev
    bb_lower = bb_basis - bb_dev
    
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
        if (np.isnan(bb_basis[i]) or 
            np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above upper BB with volume spike and price above 1d EMA34 (uptrend)
            if close[i] > bb_upper[i] and vol_spike[i] and close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Break below lower BB with volume spike and price below 1d EMA34 (downtrend)
            elif close[i] < bb_lower[i] and vol_spike[i] and close[i] < ema34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Return to opposite Bollinger Band (lower for longs, upper for shorts)
            if position == 1:
                if close[i] < bb_lower[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > bb_upper[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_Bollinger_Bands_Breakout_1dEMA34_Trend_VolumeSpike_v2"
timeframe = "4h"
leverage = 1.0