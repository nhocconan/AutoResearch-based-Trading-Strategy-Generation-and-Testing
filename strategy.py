#!/usr/bin/env python3
# 12h_donchian_breakout_1d_trend_volume_v2
# Hypothesis: Donchian channel breakout with 1d trend filter and volume confirmation on 12h timeframe.
# Long when price breaks above 20-period Donchian high with volume > 1.5x average and 1d EMA50 > EMA200.
# Short when price breaks below 20-period Donchian low with volume > 1.5x average and 1d EMA50 < EMA200.
# Exit when price returns to Donchian midpoint or opposite signal.
# Designed to capture strong trends while minimizing false breakouts in ranging markets.
# Target: 20-30 trades/year to minimize fee drag while capturing high-probability breakouts.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian_breakout_1d_trend_volume_v2"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian Channel (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_high + lowest_low) / 2
    
    # 1d EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    uptrend = ema50_1d_aligned > ema200_1d_aligned
    downtrend = ema50_1d_aligned < ema200_1d_aligned
    
    # Volume confirmation: 20-period average
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not available
        if np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or \
           np.isnan(donchian_mid[i]) or np.isnan(uptrend[i]) or \
           np.isnan(downtrend[i]) or np.isnan(avg_volume[i]):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price returns to Donchian midpoint or opposite signal
            if close[i] <= donchian_mid[i] or \
               (close[i] < lowest_low[i] and volume[i] > 1.5 * avg_volume[i] and downtrend[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price returns to Donchian midpoint or opposite signal
            if close[i] >= donchian_mid[i] or \
               (close[i] > highest_high[i] and volume[i] > 1.5 * avg_volume[i] and uptrend[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume confirmation: current volume > 1.5x average volume
            volume_ok = volume[i] > 1.5 * avg_volume[i]
            
            # Long entry: price breaks above Donchian high with volume and uptrend
            if close[i] > highest_high[i] and volume_ok and uptrend[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below Donchian low with volume and downtrend
            elif close[i] < lowest_low[i] and volume_ok and downtrend[i]:
                position = -1
                signals[i] = -0.25
    
    return signals