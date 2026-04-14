#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 4h Parabolic SAR for trend direction, 1d Williams %R for overbought/oversold conditions, and volume confirmation.
# Parabolic SAR provides clear trend-following signals with built-in acceleration factor.
# Williams %R on 1d timeframe identifies extreme levels for mean-reversion entries in the direction of the 4h trend.
# Volume confirmation (>1.3x 20-period average) filters low-quality signals.
# Designed to work in both bull and bear markets by using 4h trend direction to avoid counter-trend trades.
# Target: 25-35 trades/year per symbol (100-140 total over 4 years) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for Williams %R calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate Williams %R on 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    williams_period = 14
    highest_high = pd.Series(high_1d).rolling(window=williams_period, min_periods=williams_period).max().values
    lowest_low = pd.Series(low_1d).rolling(window=williams_period, min_periods=williams_period).min().values
    
    # Avoid division by zero
    denominator = highest_high - lowest_low
    denominator = np.where(denominator == 0, 1, denominator)
    williams_r = -100 * (highest_high - close_1d) / denominator
    
    # Load 4h data ONCE for Parabolic SAR calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Calculate Parabolic SAR on 4h data
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Parabolic SAR parameters
    af_start = 0.02
    af_increment = 0.02
    af_max = 0.2
    
    # Initialize arrays
    psar = np.zeros(len(close_4h))
    bull = np.ones(len(close_4h))  # True for bullish trend
    af = np.zeros(len(close_4h))
    ep = np.zeros(len(close_4h))  # Extreme point
    
    # Set initial values
    psar[0] = low_4h[0]
    af[0] = af_start
    ep[0] = high_4h[0]
    
    # Calculate PSAR
    for i in range(1, len(close_4h)):
        if bull[i-1]:  # Was bullish
            psar[i] = psar[i-1] + af[i-1] * (ep[i-1] - psar[i-1])
            # Check for trend reversal
            if low_4h[i] < psar[i]:
                bull[i] = False  # Reverse to bearish
                psar[i] = ep[i-1]  # SAR = prior EP
                af[i] = af_start
                ep[i] = low_4h[i]  # New EP is low
            else:
                bull[i] = True  # Stay bullish
                if high_4h[i] > ep[i-1]:
                    ep[i] = high_4h[i]  # New EP is high
                    af[i] = min(af[i-1] + af_increment, af_max)
                else:
                    ep[i] = ep[i-1]
                    af[i] = min(af[i-1] + af_increment, af_max)
        else:  # Was bearish
            psar[i] = psar[i-1] + af[i-1] * (ep[i-1] - psar[i-1])
            # Check for trend reversal
            if high_4h[i] > psar[i]:
                bull[i] = True  # Reverse to bullish
                psar[i] = ep[i-1]  # SAR = prior EP
                af[i] = af_start
                ep[i] = high_4h[i]  # New EP is high
            else:
                bull[i] = False  # Stay bearish
                if low_4h[i] < ep[i-1]:
                    ep[i] = low_4h[i]  # New EP is low
                    af[i] = min(af[i-1] + af_increment, af_max)
                else:
                    ep[i] = ep[i-1]
                    af[i] = min(af[i-1] + af_increment, af_max)
    
    # Load 4h data ONCE for volume calculation (using same 4h data for volume MA)
    vol_4h = df_4h['volume'].values
    vol_ma_4h = pd.Series(vol_4h).rolling(window=20, min_periods=20).mean().values
    
    # Align indicators to 4h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    psar_aligned = align_htf_to_ltf(prices, df_4h, psar)
    bull_aligned = align_htf_to_ltf(prices, df_4h, bull.astype(float))
    vol_ma_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_4h)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(20, 14)  # Need Williams %R period and volume MA
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(williams_r_aligned[i]) or 
            np.isnan(psar_aligned[i]) or
            np.isnan(bull_aligned[i]) or
            np.isnan(vol_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 1.3x average volume
        volume_confirmed = volume[i] > 1.3 * vol_ma_aligned[i]
        
        if position == 0:
            # Look for entries based on Williams %R extremes and 4h trend
            # Long: Williams %R oversold (< -80) AND 4h trend is bullish
            if (williams_r_aligned[i] < -80 and 
                bull_aligned[i] > 0.5 and 
                volume_confirmed):
                position = 1
                signals[i] = position_size
            # Short: Williams %R overbought (> -20) AND 4h trend is bearish
            elif (williams_r_aligned[i] > -20 and 
                  bull_aligned[i] < 0.5 and 
                  volume_confirmed):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Williams %R returns to neutral OR trend changes to bearish
            if (williams_r_aligned[i] > -50 or 
                bull_aligned[i] < 0.5):  # Trend turned bearish
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Williams %R returns to neutral OR trend changes to bullish
            if (williams_r_aligned[i] < -50 or 
                bull_aligned[i] > 0.5):  # Trend turned bullish
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_PSAR_1dWilliamsR_VolumeFilter_v1"
timeframe = "4h"
leverage = 1.0