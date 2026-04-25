#!/usr/bin/env python3
"""
12h Williams Alligator Breakout + 1d EMA34 Trend + Volume Spike
Hypothesis: Williams Alligator (jaw/teeth/lips) identifies trendless markets; 
breakouts beyond the Alligator's lips with 1d EMA34 trend filter and volume 
confirmation capture strong momentum moves. Works in bull markets (buy when 
price > lips in uptrend) and bear markets (sell when price < lips in downtrend). 
12h timeframe targets 12-37 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Williams Alligator: Smoothed Medians (Jaw=13, Teeth=8, Lips=5)
    # Smoothed Median = SMA(SMF(n), n) where SMF = (H+L+C)/3
    median_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    # Jaw: 13-period SMMA of median (8 additional smoothing)
    jaw_raw = median_price.rolling(window=13, min_periods=13).mean()
    jaw = jaw_raw.rolling(window=8, min_periods=8).mean()
    # Teeth: 8-period SMMA of median (5 additional smoothing)
    teeth_raw = median_price.rolling(window=8, min_periods=8).mean()
    teeth = teeth_raw.rolling(window=5, min_periods=5).mean()
    # Lips: 5-period SMMA of median (3 additional smoothing)
    lips_raw = median_price.rolling(window=5, min_periods=5).mean()
    lips = lips_raw.rolling(window=3, min_periods=3).mean()
    
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw.values)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth.values)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips.values)
    
    # 1d EMA34 trend filter
    ema_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for calculations
    start_idx = max(20, 13+8, 5+3)  # volume MA, Alligator components
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(lips_aligned[i]) or np.isnan(jaw_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(ema_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        vol_spike = volume_spike[i]
        
        # Trend filter: price relative to 1d EMA34
        bullish_bias = curr_close > ema_1d_aligned[i]
        bearish_bias = curr_close < ema_1d_aligned[i]
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above Alligator lips AND bullish bias AND volume spike
            long_entry = (curr_high > lips_aligned[i]) and bullish_bias and vol_spike
            # Short: price breaks below Alligator lips AND bearish bias AND volume spike
            short_entry = (curr_low < lips_aligned[i]) and bearish_bias and vol_spike
            
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
            # Exit: price falls below Alligator jaws (trend reversal) OR loss of bullish bias
            if (curr_low < jaw_aligned[i]) or (curr_close < ema_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price rises above Alligator jaws (trend reversal) OR loss of bearish bias
            if (curr_high > jaw_aligned[i]) or (curr_close > ema_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsAlligator_LipsBreakout_1dEMA34_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0