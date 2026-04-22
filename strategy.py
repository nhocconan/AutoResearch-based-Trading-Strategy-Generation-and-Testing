#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R + 1d VWAP trend filter with volume confirmation
# Williams %R identifies overbought/oversold conditions; VWAP provides dynamic trend reference
# Long when %R < -80 (oversold) + price > VWAP (uptrend) + volume spike
# Short when %R > -20 (overbought) + price < VWAP (downtrend) + volume spike
# Designed for 6h timeframe to capture mean reversion within trend with proper filtering
# Works in bull markets (buying dips in uptrend) and bear markets (selling rallies in downtrend)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for VWAP trend filter (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d VWAP (Volume Weighted Average Price)
    typical_price_1d = (high_1d + low_1d + close_1d) / 3.0
    vwap_numerator = np.cumsum(typical_price_1d * volume_1d)
    vwap_denominator = np.cumsum(volume_1d)
    vwap_1d = vwap_numerator / vwap_denominator
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    # Calculate Williams %R (14-period) on 6h data
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when high == low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Volume spike filter (20-period on 6h data)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(vwap_1d_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Williams %R < -80 (oversold) + price > VWAP (uptrend) + volume spike
            if (williams_r[i] < -80 and 
                close[i] > vwap_1d_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R > -20 (overbought) + price < VWAP (downtrend) + volume spike
            elif (williams_r[i] > -20 and 
                  close[i] < vwap_1d_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: Williams %R crosses -50 or trend reversal
            if position == 1:
                # Exit on Williams %R >= -50 or trend reversal
                if (williams_r[i] >= -50 or 
                    close[i] < vwap_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit on Williams %R <= -50 or trend reversal
                if (williams_r[i] <= -50 or 
                    close[i] > vwap_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_1dVWAP_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0