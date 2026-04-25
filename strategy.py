#!/usr/bin/env python3
"""
6h_HighLow_Band_Breakout_1dTrend_VolumeSpike
Hypothesis: On 6h timeframe, price breaks above/below a dynamic band (rolling 20-period high/low)
with 1d EMA50 trend filter and 6h volume spike confirmation (>1.5x 20-period average volume).
Targets 15-30 trades/year by requiring confluence of breakout, trend, and volume.
Works in bull markets via trend-aligned breakouts and in bear markets via mean-reversion exits
at opposing band levels. Uses discrete position sizing (0.25) to minimize fee churn.
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
    
    # 6h indicators (calculated once)
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    volume_s = pd.Series(volume)
    
    # 6h Donchian-like bands (20-period high/low)
    upper_band = high_s.rolling(window=20, min_periods=20).max().values
    lower_band = low_s.rolling(window=20, min_periods=20).min().values
    
    # 6h volume average (20-period)
    vol_avg = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # 1d data for trend filter (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for 6h bands (20) + 1d EMA50 (50)
    start_idx = max(20, 50) + 1  # Conservative warmup
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or 
            np.isnan(vol_avg[i]) or np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_vol = volume[i]
        
        # Trend filter: price relative to 1d EMA50
        uptrend = curr_close > ema_50_1d_aligned[i]
        downtrend = curr_close < ema_50_1d_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_spike = curr_vol > (1.5 * vol_avg[i])
        
        if position == 0:
            # Look for entry signals with trend and volume confirmation
            # Long breakout: price breaks above upper band with uptrend and volume spike
            long_breakout = (curr_close > upper_band[i]) and uptrend and volume_spike
            # Short breakout: price breaks below lower band with downtrend and volume spike
            short_breakout = (curr_close < lower_band[i]) and downtrend and volume_spike
            
            if long_breakout:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_breakout:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit if price breaks below lower band (mean reversion) 
            # or trend changes to downtrend
            if curr_close < lower_band[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit if price breaks above upper band (mean reversion) 
            # or trend changes to uptrend
            if curr_close > upper_band[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_HighLow_Band_Breakout_1dTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0