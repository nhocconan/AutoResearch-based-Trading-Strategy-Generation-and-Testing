#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1-day Williams %R with 20-period EMA and volume confirmation.
# Long when Williams %R crosses above -50 (bullish momentum) AND price > EMA(20) AND volume > 1.5x average.
# Short when Williams %R crosses below -50 (bearish momentum) AND price < EMA(20) AND volume > 1.5x average.
# Exit when Williams %R returns to opposite extreme (> -20 for long, < -80 for short) or volume drops below average.
# Williams %R identifies overbought/oversold conditions, EMA provides trend direction, volume confirms institutional interest.
# Designed to capture momentum swings in both bull and bear markets by trading mean reversion within the trend.
# Target: 25-40 trades/year per symbol (100-160 total over 4 years) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for Williams %R and EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough for Williams %R(14) and EMA(20)
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams %R (14)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_1d) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate EMA (20)
    ema_20 = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align indicators to 4h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    ema_20_aligned = align_htf_to_ltf(prices, df_1d, ema_20)
    
    # Volume confirmation: 1.5x average volume (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(30, 20)  # Need Williams %R and EMA periods
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(williams_r_aligned[i]) or 
            np.isnan(ema_20_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.5 * vol_ma[i]
        volume_normal = volume[i] <= vol_ma[i]  # For exit condition
        
        if position == 0:
            # Look for Williams %R cross above -50 (bullish) OR below -50 (bearish)
            # Need previous value to detect cross
            if i > 0 and not np.isnan(williams_r_aligned[i-1]):
                prev_wr = williams_r_aligned[i-1]
                curr_wr = williams_r_aligned[i]
                
                # Bullish cross: Williams %R crosses above -50
                if (prev_wr <= -50 and curr_wr > -50 and 
                    close[i] > ema_20_aligned[i] and 
                    volume_confirmed):
                    position = 1
                    signals[i] = position_size
                # Bearish cross: Williams %R crosses below -50
                elif (prev_wr >= -50 and curr_wr < -50 and 
                      close[i] < ema_20_aligned[i] and 
                      volume_confirmed):
                    position = -1
                    signals[i] = -position_size
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Williams %R returns to overbought (> -20) or volume normalizes
            if (williams_r_aligned[i] > -20 or 
                volume_normal):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Williams %R returns to oversold (< -80) or volume normalizes
            if (williams_r_aligned[i] < -80 or 
                volume_normal):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_WilliamsR_EMA20_VolumeFilter_v1"
timeframe = "4h"
leverage = 1.0