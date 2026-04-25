#!/usr/bin/env python3
"""
1d Williams %R + 1w EMA50 Trend + Volume Spike
Hypothesis: Williams %R identifies overbought/oversold conditions on daily timeframe.
In strong uptrends (price > weekly EMA50), buy when %R crosses above -80 from oversold.
In strong downtrends (price < weekly EMA50), sell when %R crosses below -20 from overbought.
Volume spike confirms institutional participation. Weekly trend filter ensures we trade
with the dominant higher-timeframe momentum, reducing false signals in choppy markets.
Designed for 1d timeframe targeting 7-25 trades/year (30-100 total over 4 years).
Works in both bull and bear markets by following weekly trend direction.
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
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Weekly EMA50
    ema_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Daily Williams %R (14-period)
    # %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    # Handle division by zero when high == low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for calculations
    start_idx = max(20, 14, 50)  # volume MA, Williams %R, weekly EMA
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(williams_r[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(ema_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        vol_spike = volume_spike[i]
        wr = williams_r[i]
        
        # Trend filter: price relative to weekly EMA50
        bullish_bias = curr_close > ema_1w_aligned[i]
        bearish_bias = curr_close < ema_1w_aligned[i]
        
        if position == 0:
            # Look for entry signals
            # Long: Williams %R crosses above -80 (oversold recovery) AND bullish bias AND volume spike
            long_entry = (wr > -80) and (williams_r[i-1] <= -80) and bullish_bias and vol_spike
            # Short: Williams %R crosses below -20 (overbought rejection) AND bearish bias AND volume spike
            short_entry = (wr < -20) and (williams_r[i-1] >= -20) and bearish_bias and vol_spike
            
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
            # Exit: Williams %R crosses above -20 (overbought) OR loss of bullish bias
            if (wr > -20) or (curr_close < ema_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: Williams %R crosses below -80 (oversold) OR loss of bearish bias
            if (wr < -80) or (curr_close > ema_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WilliamsR_1wEMA50_Trend_VolumeSpike"
timeframe = "1d"
leverage = 1.0