#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout with 1d volume confirmation and 1d trend filter
# - Donchian channel breakout (20-period) for trend following: long on upper band break, short on lower band break
# - 1d volume > 1.2x 20-period average for conviction
# - 1d EMA(50) trend filter: only take longs when price > EMA50, shorts when price < EMA50
# - Exit on opposite Donchian band (10-period) or trend reversal
# - Designed for 12h timeframe to capture multi-day trends with low trade frequency
# - Target: 15-25 trades/year to minimize fee drag

name = "12h_DonchianBreakout_1dVolume_1dTrend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for volume and trend filters
    df_1d = get_htf_data(prices, '1d')
    
    # 1d volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # 1d EMA(50) for trend direction
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Donchian channel (20-period) for entry
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max()
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min()
    donchian_upper = highest_high.values
    donchian_lower = lowest_low.values
    
    # Donchian channel (10-period) for exit
    highest_high_10 = pd.Series(high).rolling(window=10, min_periods=10).max()
    lowest_low_10 = pd.Series(low).rolling(window=10, min_periods=10).min()
    donchian_upper_10 = highest_high_10.values
    donchian_lower_10 = lowest_low_10.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]) or \
           np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or \
           np.isnan(donchian_upper_10[i]) or np.isnan(donchian_lower_10[i]):
            signals[i] = 0.0
            continue
            
        # Volume filter: current 12h volume > 1.2x 1d average volume (scaled)
        # Scale 1d average to 12h: 1d has 2x 12h bars, so divide by 2
        volume_filter = vol_ma_1d_aligned[i] > 0 and volume[i] > 1.2 * (vol_ma_1d_aligned[i] / 2.0)
        
        if position == 0:
            # Look for long entry: uptrend (price > 1d EMA50) + break above Donchian upper (20) + volume
            if close[i] > ema_50_1d_aligned[i] and close[i] > donchian_upper[i] and volume_filter:
                signals[i] = 0.25
                position = 1
            # Look for short entry: downtrend (price < 1d EMA50) + break below Donchian lower (20) + volume
            elif close[i] < ema_50_1d_aligned[i] and close[i] < donchian_lower[i] and volume_filter:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit on break below Donchian lower (10) or trend reversal
            if close[i] < donchian_lower_10[i] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit on break above Donchian upper (10) or trend reversal
            if close[i] > donchian_upper_10[i] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals