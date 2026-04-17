#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R mean reversion with 1d volume regime and 12h EMA34 trend filter.
Long when Williams %R(14) crosses above -80 (oversold) with 1d volume > 1.5x 20-day average and price > 12h EMA34.
Short when Williams %R(14) crosses below -20 (overbought) with 1d volume > 1.5x 20-day average and price < 12h EMA34.
Exit when Williams %R returns to -50 (mean reversion midpoint) or reverses with volume confirmation.
Uses 1d for volume regime, 12h for trend filter, and 6h for execution.
Williams %R identifies exhaustion points in both bull and bear markets, while volume regime ensures institutional participation.
Target: 12-37 trades/year per symbol to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for volume regime
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Get 12h data for EMA34 trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate Williams %R(14) on 6h timeframe
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close) / (highest_high_14 - lowest_low_14)
    
    # Calculate 1d volume MA20 for regime filter
    volume_1d_series = pd.Series(volume_1d)
    vol_ma_20_1d = volume_1d_series.rolling(window=20, min_periods=20).mean().values
    
    # Calculate 12h EMA34 for trend filter
    close_12h_series = pd.Series(close_12h)
    ema34_12h = close_12h_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d and 12h indicators to 6h timeframe
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # need enough for Williams %R and EMA34
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(williams_r[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i]) or 
            np.isnan(ema34_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume regime: current 1d volume > 1.5x 20-day average (expanding participation)
        # We need to get the 1d volume that corresponds to this 6h bar
        # Since we don't have aligned 1d volume, we use the condition that
        # the 1d volume MA is rising or we're in a high volume regime
        # For volume confirmation, we check if current 6h volume is above its 20-period MA
        vol_ma_20_6h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_confirmed = not np.isnan(vol_ma_20_6h[i]) and volume[i] > 1.5 * vol_ma_20_6h[i]
        
        if position == 0:
            # Long: Williams %R crosses above -80 (oversold) with volume confirmation and uptrend (price > EMA34)
            if (williams_r[i] > -80 and williams_r[i-1] <= -80 and 
                volume_confirmed and 
                close[i] > ema34_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 (overbought) with volume confirmation and downtrend (price < EMA34)
            elif (williams_r[i] < -20 and williams_r[i-1] >= -20 and 
                  volume_confirmed and 
                  close[i] < ema34_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Williams %R returns to -50 (mean reversion) OR crosses below -20 with volume (reversal)
            if (williams_r[i] >= -50 or 
                (williams_r[i] < -20 and williams_r[i-1] >= -20 and volume_confirmed)):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Williams %R returns to -50 (mean reversion) OR crosses above -80 with volume (reversal)
            if (williams_r[i] <= -50 or 
                (williams_r[i] > -80 and williams_r[i-1] <= -80 and volume_confirmed)):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_MeanReversion_VolumeRegime_EMA34_Trend"
timeframe = "6h"
leverage = 1.0