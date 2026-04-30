#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and volume spike confirmation.
# Long when price breaks above upper Donchian channel AND price > 1d EMA50 AND volume > 2.0x 20-bar average.
# Short when price breaks below lower Donchian channel AND price < 1d EMA50 AND volume > 2.0x 20-bar average.
# Exit when price crosses the Donchian midpoint (mean of upper and lower channel).
# Uses discrete position sizing (0.30) to limit drawdown and fee churn.
# Target: 75-150 total trades over 4 years (19-38/year). Works in bull/bear via 1d EMA50 trend filter.

name = "4h_Donchian20_1dEMA50_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Donchian channels (20-period) from 4h data
    # Upper = max(high, 20), Lower = min(low, 20)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    upper = high_series.rolling(window=20, min_periods=20).max().values
    lower = low_series.rolling(window=20, min_periods=20).min().values
    midpoint = (upper + lower) / 2.0  # Exit level
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # warmup for EMA and Donchian
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(midpoint[i]) or 
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: break above upper Donchian, uptrend (price > 1d EMA50), volume confirmation
            if (curr_high > upper[i] and 
                curr_close > ema_50_1d_aligned[i] and 
                curr_volume_confirm):
                signals[i] = 0.30
                position = 1
            # Short: break below lower Donchian, downtrend (price < 1d EMA50), volume confirmation
            elif (curr_low < lower[i] and 
                  curr_close < ema_50_1d_aligned[i] and 
                  curr_volume_confirm):
                signals[i] = -0.30
                position = -1
        
        elif position == 1:  # Long position
            # Exit condition: price crosses below Donchian midpoint
            if curr_close < midpoint[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:  # Short position
            # Exit condition: price crosses above Donchian midpoint
            if curr_close > midpoint[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals