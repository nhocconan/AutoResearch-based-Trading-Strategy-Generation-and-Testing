#!/usr/bin/env python3
"""
12h Williams Alligator + 1w EMA50 Trend + Volume Spike
Hypothesis: Williams Alligator (jaw=EMA13, teeth=EMA8, lips=EMA5) identifies
trend presence when lines are aligned and separated. In strong uptrends (price > 1w EMA50),
aligned Alligator with lips>teeth>jaw indicates bullish momentum for longs. In strong downtrends
(price < 1w EMA50), aligned Alligator with lips<teeth<jaw indicates bearish momentum for shorts.
Volume spike confirms institutional participation. 12h timeframe targets 12-37 trades/year
(50-150 over 4 years) to avoid fee drag while capturing major trend moves.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Williams Alligator components (using close)
    jaw = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values  # EMA13
    teeth = pd.Series(close).ewm(span=8, adjust=False, min_periods=8).mean().values   # EMA8
    lips = pd.Series(close).ewm(span=5, adjust=False, min_periods=5).mean().values    # EMA5
    
    # Alligator alignment: check if lines are separated and ordered
    bullish_align = (lips > teeth) & (teeth > jaw)  # Lips above teeth above jaw
    bearish_align = (lips < teeth) & (teeth < jaw)  # Lips below teeth below jaw
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for calculations
    start_idx = max(20, 13, 8, 5, 50)  # volume MA, Alligator EMAs, 1w EMA50 alignment
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(lips[i]) or np.isnan(teeth[i]) or np.isnan(jaw[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_bullish_align = bullish_align[i]
        curr_bearish_align = bearish_align[i]
        vol_spike = volume_spike[i]
        
        # Trend filter: price relative to 1w EMA50
        uptrend = curr_close > ema_50_aligned[i]
        downtrend = curr_close < ema_50_aligned[i]
        
        if position == 0:
            # Look for entry signals
            # Long: Bullish Alligator alignment AND uptrend AND volume spike
            long_entry = curr_bullish_align and uptrend and vol_spike
            # Short: Bearish Alligator alignment AND downtrend AND volume spike
            short_entry = curr_bearish_align and downtrend and vol_spike
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: Loss of bullish alignment OR loss of uptrend
            if (not curr_bullish_align) or (curr_close < ema_50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: Loss of bearish alignment OR loss of downtrend
            if (not curr_bearish_align) or (curr_close > ema_50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsAlligator_VolumeSpike_1wEMA50_Trend"
timeframe = "12h"
leverage = 1.0