#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R extreme reversal with 1d Elder Bull/Bear Power confluence and volume confirmation
# Williams %R identifies overbought/oversold conditions; extreme readings (<-90 or >-10) signal potential reversals.
# 1d Elder Bull Power (high - EMA13) and Bear Power (EMA13 - low) confirm underlying momentum direction.
# Volume spike (>2.0x 20-bar MA) filters for institutional participation.
# Works in bull (oversold bounces in uptrend) and bear (overbought rejections in downtrend).
# Discrete sizing (0.25) minimizes fee churn. Target: 50-150 total trades over 4 years (12-37/year).

name = "6h_WilliamsR14_Extreme_1dElderRay_Confluence_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d HTF data for Elder Ray calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d EMA(13) for Elder Ray
    ema13_1d = pd.Series(df_1d['close']).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Bull Power: high - EMA13
    bull_power_1d = df_1d['high'].values - ema13_1d
    # Elder Bear Power: EMA13 - low
    bear_power_1d = ema13_1d - df_1d['low'].values
    
    # Align Elder Ray to 6h timeframe (use prior completed 1d bar)
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # Williams %R(14) on 6h data
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close) / (highest_high_14 - lowest_low_14)
    
    # Volume confirmation: current volume > 2.0 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = 50  # Need 50 for 1d EMA13 and 20 for volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(williams_r[i]) or np.isnan(bull_power_aligned[i]) or 
            np.isnan(bear_power_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        # Williams %R extreme conditions (using current bar)
        williams_oversold = williams_r[i] <= -90  # Extreme oversold
        williams_overbought = williams_r[i] >= -10  # Extreme overbought
        
        # Elder Ray confirmation (using prior bar to avoid look-ahead)
        bull_confirmed = bull_power_aligned[i-1] > 0  # Bullish momentum
        bear_confirmed = bear_power_aligned[i-1] > 0  # Bearish momentum
        
        # Volume confirmation
        vol_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Williams %R oversold, bull power positive, volume spike
            if williams_oversold and bull_confirmed and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought, bear power positive, volume spike
            elif williams_overbought and bear_confirmed and vol_spike:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on Williams %R overbought or bear power turning positive
            if williams_r[i] >= -10 or bear_power_aligned[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on Williams %R oversold or bull power turning positive
            if williams_r[i] <= -90 or bull_power_aligned[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals