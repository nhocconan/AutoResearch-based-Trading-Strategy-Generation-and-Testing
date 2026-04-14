#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Williams Alligator (3 SMAs: Jaw=13, Teeth=8, Lips=5) for trend direction.
# Long when price > Teeth and Jaw > Teeth > Lips (bullish alignment) with volume confirmation.
# Short when price < Teeth and Jaw < Teeth < Lips (bearish alignment) with volume confirmation.
# Exit when Alligator lines cross (Teeth crosses Jaw) or price crosses Lips.
# Williams Alligator is trend-following but smooth, reducing whipsaw in choppy markets.
# Target: 15-25 trades/year per symbol (60-100 total over 4 years) to minimize fee drag.

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
    if len(df_1d) < 34:
        return np.zeros(n)
    
    median_price_1d = (df_1d['high'].values + df_1d['low'].values) / 2
    
    # Williams Alligator: Jaw (13,8), Teeth (8,5), Lips (5,3) SMAs of median price
    jaw_period = 13
    teeth_period = 8
    lips_period = 5
    
    jaw_1d = pd.Series(median_price_1d).rolling(window=jaw_period, min_periods=jaw_period).mean().values
    teeth_1d = pd.Series(median_price_1d).rolling(window=teeth_period, min_periods=teeth_period).mean().values
    lips_1d = pd.Series(median_price_1d).rolling(window=lips_period, min_periods=lips_period).mean().values
    
    # Align Alligator lines to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw_1d)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth_1d)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips_1d)
    
    # Volume confirmation: 1.5x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(jaw_period, teeth_period, lips_period, 20)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(jaw_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or
            np.isnan(lips_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        # Alligator alignment: Jaw > Teeth > Lips (bullish) or Jaw < Teeth < Lips (bearish)
        bullish_alignment = (jaw_aligned[i] > teeth_aligned[i] and 
                            teeth_aligned[i] > lips_aligned[i])
        bearish_alignment = (jaw_aligned[i] < teeth_aligned[i] and 
                            teeth_aligned[i] < lips_aligned[i])
        
        if position == 0:
            # Look for Alligator alignment entries
            # Long: price > Teeth AND bullish alignment
            if (close[i] > teeth_aligned[i] and 
                bullish_alignment and 
                volume_confirmed):
                position = 1
                signals[i] = position_size
            # Short: price < Teeth AND bearish alignment
            elif (close[i] < teeth_aligned[i] and 
                  bearish_alignment and 
                  volume_confirmed):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Teeth crosses below Jaw (Alligator sleeping) OR price crosses below Lips
            if (teeth_aligned[i] < jaw_aligned[i] or 
                close[i] < lips_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Teeth crosses above Jaw (Alligator sleeping) OR price crosses above Lips
            if (teeth_aligned[i] > jaw_aligned[i] or 
                close[i] > lips_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_WilliamsAlligator_1d_Volume_v1"
timeframe = "12h"
leverage = 1.0