#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Williams Alligator (3 SMAs) and 1w RSI trend filter.
# Long when price > Alligator Jaw (13-period SMA) and Teeth > Lips (bullish alignment) with 1w RSI > 50.
# Short when price < Alligator Jaw and Teeth < Lips (bearish alignment) with 1w RSI < 50.
# Exit when price crosses back below/above Jaw or RSI crosses 50 in opposite direction.
# Williams Alligator identifies trend phases; RSI filter avoids counter-trend trades.
# Designed for 6h timeframe to capture multi-day trends with reduced whipsaw.
# Target: 60-100 total trades over 4 years (15-25/year) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for Williams Alligator
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Williams Alligator: Jaw (13), Teeth (8), Lips (5) SMAs
    jaw = pd.Series(close_1d).rolling(window=13, min_periods=13).mean().values
    teeth = pd.Series(close_1d).rolling(window=8, min_periods=8).mean().values
    lips = pd.Series(close_1d).rolling(window=5, min_periods=5).mean().values
    
    # Bullish: price > Jaw AND Teeth > Lips
    bullish = (close_1d > jaw) & (teeth > lips)
    # Bearish: price < Jaw AND Teeth < Lips
    bearish = (close_1d < jaw) & (teeth < lips)
    
    # Load 1w data ONCE for RSI trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate RSI(14) on 1w
    delta = np.diff(close_1w, prepend=np.nan)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / avg_loss
    rs = np.where(avg_loss == 0, 100, rs)
    rsi_1w = 100 - (100 / (1 + rs))
    
    # Align indicators to lower timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    bullish_aligned = align_htf_to_ltf(prices, df_1d, bullish.astype(float))
    bearish_aligned = align_htf_to_ltf(prices, df_1d, bearish.astype(float))
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    # Volume confirmation: 1.5x average volume (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(13, 14)  # Need Alligator and RSI
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(jaw_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or
            np.isnan(lips_aligned[i]) or
            np.isnan(bullish_aligned[i]) or
            np.isnan(bearish_aligned[i]) or
            np.isnan(rsi_1w_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Look for new trend alignments
            # Long: bullish Alligator alignment AND uptrend (RSI > 50) AND volume
            if (bullish_aligned[i] == 1 and 
                rsi_1w_aligned[i] > 50 and 
                volume_confirmed):
                position = 1
                signals[i] = position_size
            # Short: bearish Alligator alignment AND downtrend (RSI < 50) AND volume
            elif (bearish_aligned[i] == 1 and 
                  rsi_1w_aligned[i] < 50 and 
                  volume_confirmed):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below Jaw OR RSI crosses below 50
            if (close[i] <= jaw_aligned[i] or 
                rsi_1w_aligned[i] <= 50):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses above Jaw OR RSI crosses above 50
            if (close[i] >= jaw_aligned[i] or 
                rsi_1w_aligned[i] >= 50):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_WilliamsAlligator_1wRSI_TrendFilter_v1"
timeframe = "6h"
leverage = 1.0