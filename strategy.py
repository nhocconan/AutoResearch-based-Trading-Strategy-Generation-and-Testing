#!/usr/bin/env python3
"""
6h Williams Alligator + 1d EMA50 Trend + Volume Spike
Hypothesis: Williams Alligator (JAW/TEETH/LIPS) identifies trendless markets; 
trade only when Alligator is "awake" (diverged) aligned with 1d EMA50 trend and volume confirmation.
In sideways markets, Alligator lines converge/trentangle → no trades (avoid whipsaw).
In trending markets, Alligator diverges → capture momentum with trend filter.
6h timeframe targets 12-37 trades/year (50-150 over 4 years).
Works in bull/bear via trend filter + avoids ranging markets via Alligator.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 1d EMA50
    ema_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Williams Alligator: SMAs of median price
    # Median price = (high + low) / 2
    median_price = (high + low) / 2.0
    
    # JAW: 13-period SMMA, shifted 8 bars
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().shift(8).values
    # TEETH: 8-period SMMA, shifted 5 bars
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().shift(5).values
    # LIPS: 5-period SMMA, shifted 3 bars
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Volume confirmation: current volume > 1.8 * 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Alligator (max shift 8 + jaw period 13)
    start_idx = max(13 + 8, 8 + 5, 5 + 3, 30, 50)  # jaw, teeth, lips, vol MA, EMA
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(ema_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        vol_spike = volume_spike[i]
        
        # Alligator "awake" condition: lines are diverged (not intertwined)
        # Bullish divergence: Lips > Teeth > JAW
        # Bearish divergence: Lips < Teeth < JAW
        bullish_diverge = (lips[i] > teeth[i]) and (teeth[i] > jaw[i])
        bearish_diverge = (lips[i] < teeth[i]) and (teeth[i] < jaw[i])
        
        # Trend filter: price relative to 1d EMA50
        bullish_bias = curr_close > ema_1d_aligned[i]
        bearish_bias = curr_close < ema_1d_aligned[i]
        
        if position == 0:
            # Look for entry signals
            # Long: bullish Alligator divergence AND bullish bias AND volume spike
            long_entry = bullish_diverge and bullish_bias and vol_spike
            # Short: bearish Alligator divergence AND bearish bias AND volume spike
            short_entry = bearish_diverge and bearish_bias and vol_spike
            
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
            # Exit: Alligator turns bearish OR loss of bullish bias
            if bearish_diverge or (curr_close < ema_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: Alligator turns bullish OR loss of bearish bias
            if bullish_diverge or (curr_close > ema_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsAlligator_1dEMA50_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0