#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d EMA50 trend + 1w volume spike filter
# In trending markets (price > 1d EMA50), we trade breakouts in trend direction: 
# long on upper Donchian breakout in uptrend, short on lower Donchian breakout in downtrend.
# Volume confirmation (>2.0x 20-period EMA of volume) reduces false breakouts.
# Designed for 4h timeframe targeting 75-200 total trades over 4 years.
# Uses discrete position sizing (0.25) to minimize fee churn and manage drawdown.

name = "4h_Donchian20_1dEMA50_1wVolumeSpike_Trend"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Get 1w data for volume spike filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 1w volume EMA20
    volume_1w = df_1w['volume'].values
    vol_ema20_1w = pd.Series(volume_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ema20_1w)
    
    # Calculate 4h Donchian channels (20-period)
    lookback = 20
    upper_channel = np.full(n, np.nan)
    lower_channel = np.full(n, np.nan)
    
    for i in range(lookback-1, n):
        upper_channel[i] = np.max(high[i-lookback+1:i+1])
        lower_channel[i] = np.min(low[i-lookback+1:i+1])
    
    # Volume confirmation: 4h volume > 2.0 x 20-period EMA of volume
    vol_ema20_4h = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ema20_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(upper_channel[i]) or 
            np.isnan(lower_channel[i]) or np.isnan(vol_ema20_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current 1w volume > 2.0 x 20-period EMA of 1w volume
        # We need current 1w volume - get the most recent completed 1w volume
        # Since we're on 4h timeframe, we approximate using aligned 1w volume EMA
        # and check if current 4h volume is elevated relative to 1w average
        volume_spike_confirm = volume[i] > (2.0 * vol_ema20_1w_aligned[i])
        
        if position == 0:
            # Determine trend: price > 1d EMA50 = uptrend, price < 1d EMA50 = downtrend
            if close[i] > ema50_1d_aligned[i]:
                # Uptrend: long on upper Donchian breakout
                if (close[i] > upper_channel[i] and 
                    volume_spike_confirm):
                    signals[i] = 0.25
                    position = 1
            else:
                # Downtrend: short on lower Donchian breakout
                if (close[i] < lower_channel[i] and 
                    volume_spike_confirm):
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: price retouches lower Donchian channel OR trend reversal
            if close[i] < lower_channel[i] or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price retouches upper Donchian channel OR trend reversal
            if close[i] > upper_channel[i] or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals