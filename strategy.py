#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation
# - Long when price breaks above Donchian(20) high with volume > 1.5x 20-period average AND daily close > daily EMA50
# - Short when price breaks below Donchian(20) low with volume > 1.5x 20-period average AND daily close < daily EMA50
# - Exit when price retreats to Donchian(10) midpoint or volume drops below average
# - Daily trend filter ensures alignment with major trend
# - Volume confirmation prevents false breakouts
# - Targets 20-50 trades/year (80-200 total over 4 years) to avoid fee drag

name = "4h_1d_donchian_breakout_volume_trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Pre-compute Donchian channels on 4h data
    high_20 = prices['high'].rolling(window=20, min_periods=20).max().values
    low_20 = prices['low'].rolling(window=20, min_periods=20).min().values
    high_10 = prices['high'].rolling(window=10, min_periods=10).max().values
    low_10 = prices['low'].rolling(window=10, min_periods=10).min().values
    donchian_mid_10 = (high_10 + low_10) / 2.0
    
    # Pre-compute volume confirmation: > 1.5x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.5 * volume_20_avg)
    
    # Pre-compute volume filter: < average volume for exit
    vol_normal = prices['volume'] < volume_20_avg
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(volume_20_avg[i]) or np.isnan(donchian_mid_10[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:  # Flat - look for new breakout entries
            # Long breakout: price > Donchian(20) high with volume spike AND daily uptrend
            if (prices['close'].iloc[i] > high_20[i] and 
                vol_spike.iloc[i] and 
                prices['close'].iloc[i] > ema50_1d_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short breakdown: price < Donchian(20) low with volume spike AND daily downtrend
            elif (prices['close'].iloc[i] < low_20[i] and 
                  vol_spike.iloc[i] and 
                  prices['close'].iloc[i] < ema50_1d_aligned[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions:
            # 1. Price retreats to Donchian(10) midpoint
            # 2. Volume drops below average (loss of momentum)
            if position == 1:  # Long position
                if (prices['close'].iloc[i] < donchian_mid_10[i] or 
                    vol_normal.iloc[i]):
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25  # Hold long
            elif position == -1:  # Short position
                if (prices['close'].iloc[i] > donchian_mid_10[i] or 
                    vol_normal.iloc[i]):
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25  # Hold short
    
    return signals