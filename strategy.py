# 4h_WeeklyDonchian_Breakout_DailyTrend_Volume
# Weekly Donchian(20) breakout with daily trend filter and volume confirmation
# Works in bull: catches breakouts; in bear: trend filter avoids false signals
# Volume confirmation reduces false breakouts
# Target: 20-50 trades/year on 4h

#!/usr/bin/env python3
name = "4h_WeeklyDonchian_Breakout_DailyTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data for Donchian channel
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly Donchian(20) - high and low of past 20 weekly candles
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate weekly Donchian channels
    high_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Align weekly Donchian levels to 4h (wait for weekly close)
    donchian_high = align_htf_to_ltf(prices, df_1w, high_20)
    donchian_low = align_htf_to_ltf(prices, df_1w, low_20)
    
    # Load daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Daily EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # ensure indicators have enough data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above weekly Donchian high + daily trend up + volume spike
            if (close[i] > donchian_high[i] and 
                close[i] > ema50_1d_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.30
                position = 1
            # Short: price breaks below weekly Donchian low + daily trend down + volume spike
            elif (close[i] < donchian_low[i] and 
                  close[i] < ema50_1d_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: price closes below weekly Donchian low
            if close[i] < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: price closes above weekly Donchian high
            if close[i] > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals