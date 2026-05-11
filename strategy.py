#!/usr/bin/env python3
"""
12h_Donchian_Breakout_20_Volume_Confirmation_Trend_Filter
Hypothesis: Donchian(20) breakout on 12h with volume confirmation and daily trend filter.
In uptrend (price > daily EMA50), go long on upper band breakout.
In downtrend (price < daily EMA50), go short on lower band breakout.
Volume > 1.5x 20-period average confirms institutional interest.
Designed for 12h timeframe to limit trades (12-37/year) while capturing strong momentum.
Works in bull markets via long breakouts and bear markets via short breakdowns.
"""

name = "12h_Donchian_Breakout_20_Volume_Confirmation_Trend_Filter"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Daily EMA 50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(
        span=50, adjust=False, min_periods=50
    ).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Donchian channels (20-period) on 12h
    # Calculate directly on 12h data (no need for HTF since we're on 12h TF)
    high_12h = high
    low_12h = low
    
    # Upper band: highest high of last 20 periods
    highest_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    # Lower band: lowest low of last 20 periods
    lowest_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 40  # Need 20 for Donchian + 20 for EMA + buffer
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ratio[i])):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume threshold
        volume_confirmed = vol_ratio[i] > 1.5
        
        if position == 0:
            # Long: break above upper Donchian + above daily EMA50 + volume
            if (close[i] > highest_high[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                volume_confirmed):
                signals[i] = 0.25
                position = 1
            # Short: break below lower Donchian + below daily EMA50 + volume
            elif (close[i] < lowest_low[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume_confirmed):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: return to opposite Donchian band or trend reversal
            if position == 1:
                # Exit long: price returns to lower band OR trend turns down
                if (close[i] <= lowest_low[i]) or \
                   (close[i] < ema_50_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price returns to upper band OR trend turns up
                if (close[i] >= highest_high[i]) or \
                   (close[i] > ema_50_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals