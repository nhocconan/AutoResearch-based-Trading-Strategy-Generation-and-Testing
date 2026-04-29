#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with weekly EMA200 trend filter and volume confirmation
# Uses daily Donchian channels for breakout signals, weekly EMA200 for trend direction
# Volume confirmation ensures breakouts have conviction
# Works in both bull and bear markets by aligning with higher timeframe trend
# Target: 15-25 trades/year (60-100 total over 4 years)

name = "1d_Donchian20_WeeklyEMA200_Trend_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop for weekly calculations
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    # Calculate weekly EMA200 for trend filter
    close_1w = df_1w['close'].values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Calculate daily Donchian channels (20-period)
    # Upper band = highest high over past 20 days
    # Lower band = lowest low over past 20 days
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(200, 20)  # warmup for EMA200 and Donchian
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if np.isnan(ema_200_aligned[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_dc_upper = donchian_upper[i]
        curr_dc_lower = donchian_lower[i]
        curr_ema200 = ema_200_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        # Determine weekly trend: price above/below EMA200
        uptrend = curr_close > curr_ema200
        downtrend = curr_close < curr_ema200
        
        if position == 0:  # Flat - look for new entries
            # Only trade with volume confirmation and in direction of weekly trend
            if curr_volume_confirm:
                # Bullish breakout: price breaks above Donchian upper in uptrend
                if uptrend and curr_close > curr_dc_upper:
                    signals[i] = 0.25
                    position = 1
                # Bearish breakout: price breaks below Donchian lower in downtrend
                elif downtrend and curr_close < curr_dc_lower:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position - exit conditions
            # Exit when: price returns to Donchian middle (median of upper/lower)
            dc_middle = (curr_dc_upper + curr_dc_lower) / 2.0
            if curr_close <= dc_middle:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position - exit conditions
            # Exit when: price returns to Donchian middle
            dc_middle = (curr_dc_upper + curr_dc_lower) / 2.0
            if curr_close >= dc_middle:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals