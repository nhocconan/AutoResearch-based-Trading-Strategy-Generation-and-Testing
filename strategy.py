#!/usr/bin/env python3
"""
6h Williams %R Reversal with 1d EMA50 Trend and Volume Spike
Hypothesis: Williams %R identifies overbought/oversold conditions. Reversals from extreme levels (%R < -80 for long, %R > -20 for short)
when aligned with 1d EMA50 trend and confirmed by volume spikes capture high-probability mean-reversion entries in both bull and bear markets.
The 6h timeframe reduces noise while the 1d EMA50 provides robust trend filtering. Volume spikes confirm institutional participation.
Designed for 12-37 trades/year (50-150 over 4 years) by requiring confluence of Williams %R reversal, 1d EMA50 trend, and volume confirmation.
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
    
    # Load 1d data ONCE before loop for Williams %R and EMA50
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Williams %R on 1d: %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    # Using 14-period lookback
    period = 14
    highest_high = pd.Series(df_1d['high']).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(df_1d['low']).rolling(window=period, min_periods=period).min().values
    williams_r = (highest_high - df_1d['close'].values) / (highest_high - lowest_low) * -100
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Align Williams %R to 6h (no extra delay needed as it's based on completed 1d bar)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # 1d EMA50 for trend filter
    ema_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume confirmation: current volume > 2.0 * 20-period average (stricter for fewer trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for calculations
    start_idx = max(20, 50, 14)  # volume MA, EMA50, Williams %R
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        vol_spike = volume_spike[i]
        wr = williams_r_aligned[i]
        
        # Trend filter: price relative to 1d EMA50
        bullish_bias = curr_close > ema_1d_aligned[i]
        bearish_bias = curr_close < ema_1d_aligned[i]
        
        if position == 0:
            # Look for entry signals - require ALL conditions: Williams %R reversal + trend + volume
            # Long: Williams %R crosses above -80 from below (oversold reversal) AND bullish bias AND volume spike
            long_entry = (wr > -80) and (williams_r_aligned[i-1] <= -80) and bullish_bias and vol_spike
            # Short: Williams %R crosses below -20 from above (overbought reversal) AND bearish bias AND volume spike
            short_entry = (wr < -20) and (williams_r_aligned[i-1] >= -20) and bearish_bias and vol_spike
            
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
            # Exit: Williams %R rises above -20 (overbought) OR loss of bullish bias
            if (wr > -20) or (curr_close < ema_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: Williams %R falls below -80 (oversold) OR loss of bearish bias
            if (wr < -80) or (curr_close > ema_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_Reversal_1dEMA50_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0