#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d volume confirmation + 1w trend filter
# - Long when price breaks above Donchian(20) high + volume > 1.5x 20d avg + price > 1w EMA(50)
# - Short when price breaks below Donchian(20) low + volume > 1.5x 20d avg + price < 1w EMA(50)
# - Exit when price crosses Donchian(10) midline or trend reverses
# - Position size: 0.25 to manage drawdown
# - Designed for 4h timeframe with weekly trend filter to work in both bull and bear markets
# - Target: 20-40 trades/year to avoid excessive fee drag

name = "4h_Donchian20_Volume_1wTrend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    
    # 1d volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # 1w EMA(50) for trend direction
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Donchian channels on 4h data
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Pre-compute session filter (08:00-20:00 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if any required data is NaN
        if np.isnan(vol_ma_1d_aligned[i]) or np.isnan(ema_50_1w_aligned[i]) or \
           np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(donchian_mid[i]):
            signals[i] = 0.0
            continue
            
        # Volume filter: current 4h volume > 1.5x 1d average volume (scaled)
        # Scale 1d average to 4h: 1d has 6 4h bars, so divide by 6
        vol_ma_4h_scaled = vol_ma_1d_aligned[i] / 6.0
        volume_filter = vol_ma_4h_scaled > 0 and volume[i] > 1.5 * vol_ma_4h_scaled
        
        if position == 0:
            # Look for long entry: price breaks above Donchian(20) high + volume + uptrend
            if close[i] > donchian_high[i] and volume_filter and close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Look for short entry: price breaks below Donchian(20) low + volume + downtrend
            elif close[i] < donchian_low[i] and volume_filter and close[i] < ema_50_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit on Donchian midline cross or trend reversal
            if close[i] < donchian_mid[i] or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit on Donchian midline cross or trend reversal
            if close[i] > donchian_mid[i] or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals