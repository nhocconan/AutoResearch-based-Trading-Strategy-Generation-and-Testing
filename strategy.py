#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 12h volume confirmation and 1d trend filter
# - Long when price breaks above Donchian(20) high with volume > 2.0x 20-bar average AND 1d close > 1d EMA50
# - Short when price breaks below Donchian(20) low with volume > 2.0x 20-bar average AND 1d close < 1d EMA50
# - Exit when price retreats to Donchian(10) midpoint OR volume drops below 0.6x average
# - Uses 1d trend filter to avoid counter-trend trades in bear markets (2025+)
# - Higher volume threshold (2.0x) and stricter exit (0.6x) reduce trade frequency (target: 15-25 trades/year)
# - Focus on BTC/ETH; SOL-only strategies are low value and will be discarded

name = "4h_12h_1d_donchian_breakout_volume_trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_12h) < 20 or len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute Donchian channels on primary timeframe (4h)
    high_20 = prices['high'].rolling(window=20, min_periods=20).max().values
    low_20 = prices['low'].rolling(window=20, min_periods=20).min().values
    high_10 = prices['high'].rolling(window=10, min_periods=10).max().values
    low_10 = prices['low'].rolling(window=10, min_periods=10).min().values
    donchian_mid_10 = (high_10 + low_10) / 2.0
    
    # Pre-compute volume confirmation: > 2.0x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (2.0 * volume_20_avg)
    
    # Pre-compute volume filter: < 0.6x average volume for exit (loss of momentum)
    vol_weak = prices['volume'] < (0.6 * volume_20_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Pre-compute aligned 12h and 1d data
    h_12h = df_12h['high'].values
    l_12h = df_12h['low'].values
    c_12h = df_12h['close'].values
    
    h_1d = df_1d['high'].values
    l_1d = df_1d['low'].values
    c_1d = df_1d['close'].values
    
    h_12h_aligned = align_htf_to_ltf(prices, df_12h, h_12h)
    l_12h_aligned = align_htf_to_ltf(prices, df_12h, l_12h)
    c_12h_aligned = align_htf_to_ltf(prices, df_12h, c_12h)
    
    h_1d_aligned = align_htf_to_ltf(prices, df_1d, h_1d)
    l_1d_aligned = align_htf_to_ltf(prices, df_1d, l_1d)
    c_1d_aligned = align_htf_to_ltf(prices, df_1d, c_1d)
    
    # Pre-compute 1d EMA(50) for trend filter
    ema50_1d = pd.Series(c_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or np.isnan(donchian_mid_10[i]) or
            np.isnan(volume_20_avg[i]) or np.isnan(ema50_1d_aligned[i]) or
            np.isnan(h_1d_aligned[i]) or np.isnan(l_1d_aligned[i]) or np.isnan(c_1d_aligned[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new breakout entries
            # Long breakout: price > Donchian(20) high with volume spike AND 1d uptrend
            if (prices['close'].iloc[i] > high_20[i] and 
                vol_spike.iloc[i] and 
                prices['close'].iloc[i] > ema50_1d_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short breakdown: price < Donchian(20) low with volume spike AND 1d downtrend
            elif (prices['close'].iloc[i] < low_20[i] and 
                  vol_spike.iloc[i] and 
                  prices['close'].iloc[i] < ema50_1d_aligned[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0  # Stay flat
        else:  # Have position - look for exit
            # Exit conditions:
            # 1. Price retreats to Donchian(10) midpoint
            # 2. Volume drops below 0.6x average (loss of momentum)
            if position == 1:  # Long position
                if (prices['close'].iloc[i] < donchian_mid_10[i] or 
                    vol_weak.iloc[i]):
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25  # Hold long
            elif position == -1:  # Short position
                if (prices['close'].iloc[i] > donchian_mid_10[i] or 
                    vol_weak.iloc[i]):
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25  # Hold short
    
    return signals