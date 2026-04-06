#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout + 1d daily trend filter + volume confirmation
# Long when price breaks above 20-period high AND daily EMA50 > EMA200 AND volume > 1.5x average
# Short when price breaks below 20-period low AND daily EMA50 < EMA200 AND volume > 1.5x average
# Exit when price crosses opposite Donchian level or volume dries up
# Works in bull markets via breakouts and bear markets via breakdowns
# Targets 50-150 total trades over 4 years (12-37/year)

name = "6h_donchian20_1d_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian Channel (20-period) - breakout signals
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max()
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min()
    donchian_high = highest_high.values
    donchian_low = lowest_low.values
    
    # Daily trend filter: EMA50 vs EMA200
    df_1d = get_htf_data(prices, '1d')
    daily_close = df_1d['close'].values
    
    # Calculate EMA50 and EMA200 on daily close
    ema50 = pd.Series(daily_close).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema200 = pd.Series(daily_close).ewm(span=200, min_periods=200, adjust=False).mean().values
    
    # Align daily EMAs to 6h timeframe
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50)
    ema200_aligned = align_htf_to_ltf(prices, df_1d, ema200)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_threshold = 1.5 * volume_ma.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(ema50_aligned[i]) or np.isnan(ema200_aligned[i]) or
            np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Exit conditions: price crosses opposite Donchian level OR volume drops
        if position == 1:  # long position
            if close[i] <= donchian_low[i] or volume[i] < volume_threshold[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if close[i] >= donchian_high[i] or volume[i] < volume_threshold[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for breakouts with trend alignment and volume confirmation
            # Bullish breakout: price above Donchian high + daily uptrend + volume
            if (close[i] > donchian_high[i] and 
                ema50_aligned[i] > ema200_aligned[i] and
                volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Bearish breakdown: price below Donchian low + daily downtrend + volume
            elif (close[i] < donchian_low[i] and
                  ema50_aligned[i] < ema200_aligned[i] and
                  volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
    
    return signals