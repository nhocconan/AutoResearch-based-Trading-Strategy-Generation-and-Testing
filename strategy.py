#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d EMA34 filter and volume spike confirmation.
# Williams Alligator: Jaw (EMA13, 8 shift), Teeth (EMA8, 5 shift), Lips (EMA5, 3 shift).
# Long: Lips > Teeth > Jaw (bullish alignment) + price above 1d EMA34 + volume spike.
# Short: Lips < Teeth < Jaw (bearish alignment) + price below 1d EMA34 + volume spike.
# Uses 12h timeframe to reduce trade frequency; 1d EMA34 for trend filter; volume spike for confirmation.
# Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag.
name = "12h_WilliamsAlligator_1dEMA34_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA34 filter (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Williams Alligator components
    close_s = pd.Series(close)
    lips = close_s.ewm(span=5, adjust=False, min_periods=5).mean().shift(3).values  # EMA5, 3 shift
    teeth = close_s.ewm(span=8, adjust=False, min_periods=8).mean().shift(5).values  # EMA8, 5 shift
    jaw = close_s.ewm(span=13, adjust=False, min_periods=13).mean().shift(8).values  # EMA13, 8 shift
    
    # 1d EMA34 for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume spike: current volume > 2.0 * 20-period average volume
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(lips[i]) or np.isnan(teeth[i]) or np.isnan(jaw[i]) or
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 1d EMA34
        uptrend = close[i] > ema34_1d_aligned[i]
        downtrend = close[i] < ema34_1d_aligned[i]
        
        # Alligator alignment
        bullish_alignment = lips[i] > teeth[i] and teeth[i] > jaw[i]
        bearish_alignment = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        if position == 0:
            # Long: bullish alignment AND uptrend AND volume spike
            if bullish_alignment and uptrend and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: bearish alignment AND downtrend AND volume spike
            elif bearish_alignment and downtrend and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: bullish alignment breaks OR trend reverses
            if not bullish_alignment or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: bearish alignment breaks OR trend reverses
            if not bearish_alignment or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals