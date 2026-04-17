#/usr/bin/env python3
"""
1d_Weekly_Donchian20_VolumeTrend_v2
Hypothesis: Weekly Donchian breakout on 1d timeframe with volume confirmation and 1w EMA trend filter.
Long when price breaks above weekly Donchian upper band in uptrend (close > weekly EMA50),
short when breaks below lower band in downtrend (close < weekly EMA50).
Volume must be above 1.5x 20-period weekly average. Position size 0.25.
Designed for low trade frequency (<25/year) to minimize fee drag and work in both bull/bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1w data for trend filter and Donchian channels ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    volume_1w = df_1w['volume'].values
    
    # 1w EMA50 for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # 1w Donchian channel (20)
    high_series = pd.Series(high_1w)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1w, donchian_upper)
    
    low_series = pd.Series(low_1w)
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1w, donchian_lower)
    
    # 1w volume average (20-period)
    vol_avg20_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    vol_avg20_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_avg20_1w)
    
    signals = np.zeros(n)
    
    # Warmup: covers EMA50, Donchian, and volume average
    warmup = 60
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(donchian_upper_aligned[i]) or 
            np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(vol_avg20_1w_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Get current 1w volume
        vol_1w_current = align_htf_to_ltf(prices, df_1w, volume_1w)[i]
        
        # Volume filter: current volume > 1.5x 20-period average
        vol_filter = vol_1w_current > 1.5 * vol_avg20_1w_aligned[i]
        
        # Entry conditions: Donchian breakout in trend direction
        if position == 0:
            # Long: break above upper band in uptrend (close > EMA50) with volume
            if close[i] > donchian_upper_aligned[i] and close[i] > ema50_1w_aligned[i] and vol_filter:
                signals[i] = 0.25
                position = 1
                continue
            # Short: break below lower band in downtrend (close < EMA50) with volume
            elif close[i] < donchian_lower_aligned[i] and close[i] < ema50_1w_aligned[i] and vol_filter:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit conditions: reverse when price crosses opposite Donchian band
        elif position == 1:
            if close[i] < donchian_lower_aligned[i]:  # exit long when price breaks below lower band
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            if close[i] > donchian_upper_aligned[i]:  # exit short when price breaks above upper band
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Weekly_Donchian20_VolumeTrend_v2"
timeframe = "1d"
leverage = 1.0