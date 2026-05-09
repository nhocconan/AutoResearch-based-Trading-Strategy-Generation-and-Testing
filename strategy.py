#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator with 1d trend filter and volume spike filter.
# Uses Williams Alligator (3 SMAs: Jaw=13, Teeth=8, Lips=5) to identify trends.
# Long when Lips > Teeth > Jaw with 1d uptrend and volume spike.
# Short when Lips < Teeth < Jaw with 1d downtrend and volume spike.
# Designed to work in both bull (follow 1d uptrend) and bear (follow 1d downtrend) markets.
# Target: 20-50 trades/year to avoid fee drag.
name = "4h_Williams_Alligator_1dTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    # Williams Alligator: 3 SMAs on median price
    median_price = (high + low) / 2
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().values  # Blue line
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().values    # Red line
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().values     # Green line
    
    # Align Alligator lines to 4h timeframe
    jaw_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), jaw)
    teeth_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), teeth)
    lips_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), lips)
    
    # 1d EMA(34) for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: volume > 2.0x 20-period EMA (high threshold for fewer trades)
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (2.0 * vol_ema20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 13  # Need enough data for Alligator jaws
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(lips_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(jaw_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Enter long: Lips > Teeth > Jaw (bullish alignment) + 1d uptrend + volume spike
            if (lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i] and 
                price > ema_34_1d_aligned[i] and vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: Lips < Teeth < Jaw (bearish alignment) + 1d downtrend + volume spike
            elif (lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i] and 
                  price < ema_34_1d_aligned[i] and vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Alligator alignment breaks or trend reverses
            if not (lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i]) or price < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Alligator alignment breaks or trend reverses
            if not (lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i]) or price > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals