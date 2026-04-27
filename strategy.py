#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h 4-hour Donchian breakout with 1-day trend filter and volume confirmation
# Uses Donchian channel breakouts (20-period) from 4h timeframe for directional bias
# Enters on 1h timeframe when price breaks above/below 4h Donchian bands with volume confirmation
# Filters trades to 08-20 UTC session to reduce noise
# Designed for 15-35 trades/year per symbol to minimize fee drag while maintaining edge

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Donchian channel calculation (directional bias)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate 20-period Donchian channels on 4h data
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian channels to 1h timeframe (only use completed 4h bars)
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # Get 1d data for trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: current volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.3)
    
    # Session filter: 08-20 UTC (reduces noise trading)
    session_mask = np.zeros(n, dtype=bool)
    for i in range(n):
        hour = pd.Timestamp(prices['open_time'].iloc[i]).hour
        session_mask[i] = 8 <= hour <= 20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma[i]) or
            not session_mask[i]):
            signals[i] = 0.0
            continue
        
        # Long conditions: price breaks above 4h Donchian high AND 1d uptrend AND volume
        if (close[i] > donchian_high_aligned[i] and 
            close[i] > ema50_1d_aligned[i] and 
            volume_filter[i]):
            signals[i] = 0.20
            position = 1
        # Short conditions: price breaks below 4h Donchian low AND 1d downtrend AND volume
        elif (close[i] < donchian_low_aligned[i] and 
              close[i] < ema50_1d_aligned[i] and 
              volume_filter[i]):
            signals[i] = -0.20
            position = -1
        else:
            # Hold current position or flat
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
    
    return signals

name = "1h_DonchianBreakout_4hDirection_1dTrend_Volume"
timeframe = "1h"
leverage = 1.0