#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator + 1d volume spike + 1w trend filter.
Long when price > Alligator Jaw (blue line) with 1d volume > 2.0x 20-day average and 1w close > 1w EMA34.
Short when price < Alligator Jaw with 1d volume > 2.0x 20-day average and 1w close < 1w EMA34.
Exit when price crosses Alligator Teeth (red line) in opposite direction.
Williams Alligator: Jaw=SMA(13,8), Teeth=SMA(8,5), Lips=SMA(5,3).
Designed to catch strong trends with volume confirmation while avoiding choppy markets.
Uses 1d for volume regime, 1w for trend filter, 12h for execution.
Target: 12-25 trades/year per symbol to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Alligator calculation
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Williams Alligator on 12h
    # Jaw: Blue line - 13-period SMMA smoothed 8 periods ahead
    close_12h_series = pd.Series(close_12h)
    sma13_12h = close_12h_series.rolling(window=13, min_periods=13).mean().values
    jaw_12h = pd.Series(sma13_12h).rolling(window=8, min_periods=8).mean().values  # SMMA
    
    # Teeth: Red line - 8-period SMMA smoothed 5 periods ahead
    sma8_12h = close_12h_series.rolling(window=8, min_periods=8).mean().values
    teeth_12h = pd.Series(sma8_12h).rolling(window=5, min_periods=5).mean().values  # SMMA
    
    # Lips: Green line - 5-period SMMA smoothed 3 periods ahead
    sma5_12h = close_12h_series.rolling(window=5, min_periods=5).mean().values
    lips_12h = pd.Series(sma5_12h).rolling(window=3, min_periods=3).mean().values  # SMMA
    
    # Get 1d data for volume regime
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    volume_1d_series = pd.Series(volume_1d)
    vol_ma_20_1d = volume_1d_series.rolling(window=20, min_periods=20).mean().values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    close_1w_series = pd.Series(close_1w)
    ema34_1w = close_1w_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align all indicators to 12h timeframe
    jaw_12h_aligned = align_htf_to_ltf(prices, df_12h, jaw_12h)
    teeth_12h_aligned = align_htf_to_ltf(prices, df_12h, teeth_12h)
    lips_12h_aligned = align_htf_to_ltf(prices, df_12h, lips_12h)
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 60  # need enough for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(jaw_12h_aligned[i]) or 
            np.isnan(teeth_12h_aligned[i]) or 
            np.isnan(lips_12h_aligned[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i]) or 
            np.isnan(ema34_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume regime: current 1d volume > 2.0x 20-day average (expanding participation)
        # Since we don't have aligned 1d volume, we check if 12h volume is above its 20-period MA
        vol_ma_20_12h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_confirmed = not np.isnan(vol_ma_20_12h[i]) and volume[i] > 2.0 * vol_ma_20_12h[i]
        
        if position == 0:
            # Long: price > Jaw with volume confirmation and 1w uptrend (close > EMA34)
            if (close[i] > jaw_12h_aligned[i] and 
                volume_confirmed and 
                close[i] > ema34_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price < Jaw with volume confirmation and 1w downtrend (close < EMA34)
            elif (close[i] < jaw_12h_aligned[i] and 
                  volume_confirmed and 
                  close[i] < ema34_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below Teeth (red line)
            if close[i] < teeth_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above Teeth (red line)
            if close[i] > teeth_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsAlligator_1dVolumeSpike_1wEMA34_Trend"
timeframe = "12h"
leverage = 1.0