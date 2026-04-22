#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator with 1d trend filter and volume confirmation
# Uses Williams Alligator (SMAs of median price) to identify trend direction and strength.
# Long when jaw < teeth < lips (bullish alignment) and price above teeth.
# Short when jaw > teeth > lips (bearish alignment) and price below teeth.
# Filtered by 1d EMA50 trend and confirmed by volume spikes (>1.8x 20-period average).
# Designed for 4h timeframe with moderate entry frequency to avoid overtrading (<300 total trades).
# Works in both bull and bear markets by requiring alignment with daily trend.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for trend filter (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams Alligator components (based on median price)
    median_price = (high + low) / 2
    
    # Jaw (blue line): 13-period SMMA, shifted 8 bars forward
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().values
    jaw = np.roll(jaw, 8)  # shift forward
    jaw[:8] = np.nan  # first 8 values invalid
    
    # Teeth (red line): 8-period SMMA, shifted 5 bars forward
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().values
    teeth = np.roll(teeth, 5)  # shift forward
    teeth[:5] = np.nan  # first 5 values invalid
    
    # Lips (green line): 5-period SMMA, shifted 3 bars forward
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().values
    lips = np.roll(lips, 3)  # shift forward
    lips[:3] = np.nan  # first 3 values invalid
    
    # Align Alligator lines to 4h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bullish alignment (jaw < teeth < lips) + price above teeth + daily uptrend + volume spike
            if (jaw_aligned[i] < teeth_aligned[i] and 
                teeth_aligned[i] < lips_aligned[i] and
                close[i] > teeth_aligned[i] and
                close[i] > ema_50_1d_aligned[i] and
                volume[i] > 1.8 * vol_avg_20[i]):
                signals[i] = 0.25
                position = 1
            # Short: Bearish alignment (jaw > teeth > lips) + price below teeth + daily downtrend + volume spike
            elif (jaw_aligned[i] > teeth_aligned[i] and 
                  teeth_aligned[i] > lips_aligned[i] and
                  close[i] < teeth_aligned[i] and
                  close[i] < ema_50_1d_aligned[i] and
                  volume[i] > 1.8 * vol_avg_20[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: when Alligator lines cross or trend reverses
            if position == 1:
                # Exit long: bearish crossover or price below teeth or trend turns down
                if (jaw_aligned[i] > teeth_aligned[i] or 
                    close[i] < teeth_aligned[i] or
                    close[i] < ema_50_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short: bullish crossover or price above teeth or trend turns up
                if (jaw_aligned[i] < teeth_aligned[i] or 
                    close[i] > teeth_aligned[i] or
                    close[i] > ema_50_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_WilliamsAlligator_1dEMA50_VolumeConfirmation"
timeframe = "4h"
leverage = 1.0