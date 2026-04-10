#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h trend filter and volume confirmation
# - Long when price breaks above 4h Donchian high with volume > 1.8x average AND 12h close > 12h EMA34
# - Short when price breaks below 4h Donchian low with volume > 1.8x average AND 12h close < 12h EMA34
# - Exit when price retests 4h Donchian midpoint or volume drops below average
# - 12h EMA34 trend filter ensures alignment with intermediate trend
# - Volume confirmation prevents false breakouts
# - Targets 20-40 trades/year (80-160 total over 4 years) to avoid fee drag
# - Donchian breakouts work well in both trending and ranging markets when combined with trend and volume filters

name = "4h_12h_donchian_breakout_volume_trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    # Pre-compute Donchian channels (20-period) on 4h data
    high_20 = prices['high'].rolling(window=20, min_periods=20).max().values
    low_20 = prices['low'].rolling(window=20, min_periods=20).min().values
    midpoint_20 = (high_20 + low_20) / 2.0
    
    # Pre-compute 12h EMA(34) for trend filter
    close_12h = df_12h['close'].values
    ema34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    # Pre-compute volume confirmation: > 1.8x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.8 * volume_20_avg)
    
    # Pre-compute volume filter: < average volume for exit
    vol_normal = prices['volume'] < volume_20_avg
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(34, n):
        # Skip if any required data is invalid
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(midpoint_20[i]) or np.isnan(ema34_12h_aligned[i]) or
            np.isnan(volume_20_avg[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:  # Flat - look for new breakout entries
            # Long breakout: price > 4h Donchian high with volume spike AND 12h uptrend
            if (prices['close'].iloc[i] > high_20[i] and 
                vol_spike.iloc[i] and 
                prices['close'].iloc[i] > ema34_12h_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short breakdown: price < 4h Donchian low with volume spike AND 12h downtrend
            elif (prices['close'].iloc[i] < low_20[i] and 
                  vol_spike.iloc[i] and 
                  prices['close'].iloc[i] < ema34_12h_aligned[i]):
                position = -1
                signals[i] = -0.25
        else:  # Have position - look for exit
            # Exit conditions:
            # 1. Price retests 4h Donchian midpoint (mean reversion signal)
            # 2. Volume drops below average (loss of momentum)
            if position == 1:  # Long position
                if (prices['close'].iloc[i] < midpoint_20[i] or 
                    vol_normal.iloc[i]):
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25  # Hold long
            elif position == -1:  # Short position
                if (prices['close'].iloc[i] > midpoint_20[i] or 
                    vol_normal.iloc[i]):
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25  # Hold short
    
    return signals