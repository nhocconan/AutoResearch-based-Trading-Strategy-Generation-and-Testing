#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w trend filter and volume confirmation
# - Long when price breaks above 20-day high with volume > 1.5x average AND weekly close > weekly EMA20
# - Short when price breaks below 20-day low with volume > 1.5x average AND weekly close < weekly EMA20
# - Exit when price retests 10-day midpoint or volume drops below average
# - Weekly trend filter ensures alignment with major trend
# - Volume confirmation prevents false breakouts
# - Targets 15-25 trades/year (60-100 total over 4 years) to avoid fee drag
# - Donchian breakouts work well in both trending and ranging markets when combined with trend and volume filters

name = "1d_1w_donchian_breakout_volume_trend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Pre-compute Donchian channels (20-period) on daily data
    high_20 = prices['high'].rolling(window=20, min_periods=20).max().values
    low_20 = prices['low'].rolling(window=20, min_periods=20).min().values
    # 10-period midpoint for exit
    high_10 = prices['high'].rolling(window=10, min_periods=10).max().values
    low_10 = prices['low'].rolling(window=10, min_periods=10).min().values
    midpoint_10 = (high_10 + low_10) / 2.0
    
    # Pre-compute 1w EMA(20) for trend filter
    close_1w = df_1w['close'].values
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Pre-compute volume confirmation: > 1.5x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.5 * volume_20_avg)
    
    # Pre-compute volume filter: < average volume for exit
    vol_normal = prices['volume'] < volume_20_avg
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(midpoint_10[i]) or np.isnan(ema20_1w_aligned[i]) or
            np.isnan(volume_20_avg[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:  # Flat - look for new breakout entries
            # Long breakout: price > 20-day high with volume spike AND weekly uptrend
            if (prices['close'].iloc[i] > high_20[i] and 
                vol_spike.iloc[i] and 
                prices['close'].iloc[i] > ema20_1w_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short breakdown: price < 20-day low with volume spike AND weekly downtrend
            elif (prices['close'].iloc[i] < low_20[i] and 
                  vol_spike.iloc[i] and 
                  prices['close'].iloc[i] < ema20_1w_aligned[i]):
                position = -1
                signals[i] = -0.25
        else:  # Have position - look for exit
            # Exit conditions:
            # 1. Price retests 10-day midpoint (mean reversion signal)
            # 2. Volume drops below average (loss of momentum)
            if position == 1:  # Long position
                if (prices['close'].iloc[i] < midpoint_10[i] or 
                    vol_normal.iloc[i]):
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25  # Hold long
            elif position == -1:  # Short position
                if (prices['close'].iloc[i] > midpoint_10[i] or 
                    vol_normal.iloc[i]):
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25  # Hold short
    
    return signals