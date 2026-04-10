#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation
# - Long when price breaks above Donchian upper (20-period high) with volume > 1.8x average AND 1d close > 1d EMA50
# - Short when price breaks below Donchian lower (20-period low) with volume > 1.8x average AND 1d close < 1d EMA50
# - Exit when price retreats to Donchian midpoint OR volume drops below 0.9x average
# - Uses 1d trend filter to avoid counter-trend trades in bear markets (2025+)
# - Moderate volume threshold (1.8x) reduces false breakouts while maintaining sufficient trades
# - Donchian midpoint exit provides symmetric risk/reward
# - Target: 25-40 trades/year (100-160 total over 4 years) to minimize fee drag

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
    
    # Pre-compute Donchian channels (20-period) on 4h data
    high_20_max = prices['high'].rolling(window=20, min_periods=20).max().values
    low_20_min = prices['low'].rolling(window=20, min_periods=20).min().values
    donchian_mid = (high_20_max + low_20_min) / 2.0
    
    # Pre-compute volume confirmation: > 1.8x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.8 * volume_20_avg)
    
    # Pre-compute volume filter: < 0.9x average volume for exit (loss of momentum)
    vol_weak = prices['volume'] < (0.9 * volume_20_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(high_20_max[i]) or 
            np.isnan(low_20_min[i]) or np.isnan(donchian_mid[i]) or 
            np.isnan(volume_20_avg[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new breakout entries
            # Long breakout: price > Donchian upper with volume spike AND 1d uptrend
            if (prices['close'].iloc[i] > high_20_max[i] and 
                vol_spike[i] and 
                prices['close'].iloc[i] > ema50_1d_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short breakdown: price < Donchian lower with volume spike AND 1d downtrend
            elif (prices['close'].iloc[i] < low_20_min[i] and 
                  vol_spike[i] and 
                  prices['close'].iloc[i] < ema50_1d_aligned[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0  # Stay flat
        else:  # Have position - look for exit
            # Exit conditions:
            # 1. Price retreats to Donchian midpoint
            # 2. Volume drops below 0.9x average (loss of momentum)
            if position == 1:  # Long position
                if (prices['close'].iloc[i] < donchian_mid[i] or 
                    vol_weak[i]):
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25  # Hold long
            elif position == -1:  # Short position
                if (prices['close'].iloc[i] > donchian_mid[i] or 
                    vol_weak[i]):
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25  # Hold short
    
    return signals