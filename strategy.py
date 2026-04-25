#!/usr/bin/env python3
"""
1h Williams %R Mean Reversion with 4h EMA50 Trend Filter and Volume Spike
Hypothesis: Williams %R identifies overbought/oversold conditions for mean reversion entries.
4h EMA50 provides higher timeframe trend bias to avoid counter-trend trades.
Volume spike confirms momentum behind the reversal. Session filter (08-20 UTC) reduces noise.
Designed for 1h timeframe targeting 15-35 trades/year. Works in bull/bear via trend filter.
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
    
    # Precompute session hours (08-20 UTC) - open_time is already datetime64[ms]
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 4h data ONCE before loop for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Williams %R (14) on 1h
    period = 14
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # Volume confirmation: current volume > 1.8 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for all indicators
    start_idx = max(period, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready or outside session
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(vol_ma[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        wr = williams_r[i]
        vol_spike = volume_spike[i]
        
        # Trend filter: price relative to 4h EMA50
        bullish_bias = curr_close > ema_4h_aligned[i]
        bearish_bias = curr_close < ema_4h_aligned[i]
        
        if position == 0:
            # Look for entry signals - require: Williams %R extreme + trend + volume
            # Long: Williams %R < -80 (oversold) AND bullish bias AND volume spike
            long_entry = (wr < -80) and bullish_bias and vol_spike
            # Short: Williams %R > -20 (overbought) AND bearish bias AND volume spike
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
            # Exit: Williams %R rises above -50 (momentum fading) OR loss of bullish bias
            if (wr > -50) or (curr_close < ema_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short position management
            # Exit: Williams %R falls below -50 (momentum fading) OR loss of bearish bias
            if (wr < -50) or (curr_close > ema_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_WilliamsR_MeanReversion_4hEMA50_Trend_VolumeSpike_Session"
timeframe = "1h"
leverage = 1.0