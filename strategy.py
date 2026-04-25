#!/usr/bin/env python3
"""
4h Williams Alligator with 1d EMA50 Trend Filter and Volume Spike
Hypothesis: Williams Alligator (jaw-teeth-lips) identifies trendless markets when lines are intertwined.
Trend emerges when lines diverge in alignment with price. Combined with 1d EMA50 trend filter and volume
confirmation, this captures strong trending moves while avoiding chop. Designed for 4h timeframe to target
20-50 trades/year by requiring confluence of Alligator signals, 1d EMA50 trend, and volume confirmation.
Works in both bull (long when lips>teeth>jaw and price above) and bear (short when lips<teeth<jaw and price below).
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
    
    # Load 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    ema_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Williams Alligator components on primary timeframe (4h)
    # Jaw: 13-period SMMA shifted 8 bars forward
    # Teeth: 8-period SMMA shifted 5 bars forward  
    # Lips: 5-period SMMA shifted 3 bars forward
    # SMMA (Smoothed Moving Average) approximation using EMA with adjusted alpha
    close_series = pd.Series(close)
    
    # Jaw (13, 8)
    jaw = close_series.ewm(alpha=1/13, adjust=False).mean().shift(8).values
    # Teeth (8, 5)
    teeth = close_series.ewm(alpha=1/8, adjust=False).mean().shift(5).values
    # Lips (5, 3)
    lips = close_series.ewm(alpha=1/5, adjust=False).mean().shift(3).values
    
    # Volume confirmation: current volume > 1.8 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Alligator (max shift 8) and EMA50
    start_idx = max(13, 50)  # Alligator jaw period, EMA50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        vol_spike = volume_spike[i]
        
        # Alligator conditions: lines not intertwined (trending)
        # Bullish alignment: Lips > Teeth > Jaw
        # Bearish alignment: Lips < Teeth < Jaw
        bullish_alligator = (lips[i] > teeth[i]) and (teeth[i] > jaw[i])
        bearish_alligator = (lips[i] < teeth[i]) and (teeth[i] < jaw[i])
        
        # Trend filter: price relative to 1d EMA50
        bullish_bias = curr_close > ema_1d_aligned[i]
        bearish_bias = curr_close < ema_1d_aligned[i]
        
        if position == 0:
            # Look for entry signals - require ALL conditions: Alligator alignment + trend + volume
            # Long: bullish Alligator alignment AND bullish bias AND volume spike
            long_entry = bullish_alligator and bullish_bias and vol_spike
            # Short: bearish Alligator alignment AND bearish bias AND volume spike
            short_entry = bearish_alligator and bearish_bias and vol_spike
            
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
            # Exit: Alligator lines re-intertwine (chop) OR loss of bullish bias
            if not bullish_alligator or (curr_close < ema_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: Alligator lines re-intertwine (chop) OR loss of bearish bias
            if not bearish_alligator or (curr_close > ema_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WilliamsAlligator_1dEMA50_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0