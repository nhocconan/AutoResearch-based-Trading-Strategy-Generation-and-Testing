#!/usr/bin/env python3
"""
1h Williams %R + 4h EMA34 Trend + Volume Spike
Hypothesis: Williams %R captures short-term reversals in overextended markets. 
4h EMA34 provides institutional trend bias. Volume spike confirms participation.
Works in both bull/bear markets by only taking trades aligned with higher timeframe trend.
1h timeframe with strict entry conditions targets 15-37 trades/year.
Uses session filter (08-20 UTC) to reduce noise and avoid low-volume periods.
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
    
    # Load 4h data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 35:
        return np.zeros(n)
    
    # 4h EMA34
    ema_4h = pd.Series(df_4h['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Williams %R (14-period) on 1h
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Volume confirmation: current volume > 1.8 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.8)
    
    # Session filter: 08-20 UTC (precomputed for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for calculations
    start_idx = max(14, 20, 34)
    
    for i in range(start_idx, n):
        # Skip if not in trading session or data not ready
        if not in_session[i] or \
           np.isnan(ema_4h_aligned[i]) or np.isnan(williams_r[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        vol_spike = volume_spike[i]
        wr = williams_r[i]
        
        # Trend filter: price relative to 4h EMA34
        bullish_bias = curr_close > ema_4h_aligned[i]
        bearish_bias = curr_close < ema_4h_aligned[i]
        
        if position == 0:
            # Look for entry signals
            # Long: Williams %R oversold (< -80) AND bullish bias AND volume spike
            long_entry = (wr < -80) and bullish_bias and vol_spike
            # Short: Williams %R overbought (> -20) AND bearish bias AND volume spike
            short_entry = (wr > -20) and bearish_bias and vol_spike
            
            if long_entry:
                signals[i] = 0.20
                position = 1
            elif short_entry:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: Williams %R returns above -50 (momentum fading) OR loss of bullish bias
            if (wr > -50) or (curr_close < ema_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short position management
            # Exit: Williams %R returns below -50 (momentum fading) OR loss of bearish bias
            if (wr < -50) or (curr_close > ema_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_WilliamsR_4hEMA34_Trend_VolumeSpike_Session"
timeframe = "1h"
leverage = 1.0