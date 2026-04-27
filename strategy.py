#!/usr/bin/env python3
"""
12h_Donchian20_Breakout_1wTrend_Volume_Spike_v1
Hypothesis: Uses 12h Donchian(20) breakouts filtered by 1w trend (EMA50) and volume spikes.
In bull markets, rides trends; in bear markets, captures short-term reversals after volatility spikes.
Volume confirmation reduces false breakouts. Designed for low trade frequency (15-30/year) to minimize fee drag.
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
    
    # Calculate 12h Donchian channels
    donchian_period = 20
    donchian_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # 1w trend filter: EMA50
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for Donchian(20) and EMA50
    start_idx = max(donchian_period, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        upper = donchian_high[i]
        lower = donchian_low[i]
        ema50 = ema50_1w_aligned[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Determine trend: price vs EMA50 (1w)
            uptrend = close_val > ema50
            downtrend = close_val < ema50
            
            if uptrend and vol_conf:
                # Long: break above upper Donchian with volume
                if close_val > upper:
                    signals[i] = size
                    position = 1
                    entry_price = close_val
            elif downtrend and vol_conf:
                # Short: break below lower Donchian with volume
                if close_val < lower:
                    signals[i] = -size
                    position = -1
                    entry_price = close_val
        elif position == 1:
            # Exit: price re-enters below lower Donchian or trend reversal
            if close_val < lower:  # Re-enter below lower band
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit: price re-enters above upper Donchian or trend reversal
            if close_val > upper:  # Re-enter above upper band
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Donchian20_Breakout_1wTrend_Volume_Spike_v1"
timeframe = "12h"
leverage = 1.0