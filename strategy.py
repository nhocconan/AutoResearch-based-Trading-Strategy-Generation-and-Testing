#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with weekly trend filter (1w EMA50) and 1d volume spike confirmation
# Long when: price > Donchian upper(20) AND price > 1w EMA50 AND 1d volume > 2x 20-day avg volume
# Short when: price < Donchian lower(20) AND price < 1w EMA50 AND 1d volume > 2x 20-day avg volume
# Exit when: price crosses Donchian midpoint OR volume drops below threshold
# Weekly trend filter prevents counter-trend trades in ranging markets
# Volume spike confirms institutional participation in breakouts
# Target: 12-30 trades/year via tight entry conditions requiring confluence of 3 filters

name = "6h_Donchian20_1wEMA50_Trend_1dVolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for volume analysis
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate 1d volume 20-bar moving average
    vol_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d = np.concatenate([np.full(19, np.nan), vol_ma_20_1d])  # Align with 1d bars
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w = np.concatenate([np.full(49, np.nan), ema_50_1w])  # Align with 1w bars
    
    # Align 1d and 1w indicators to 6h timeframe
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Donchian channels on 6h data (20-bar period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_upper + donchian_lower) / 2
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Need sufficient history for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        vol_spike = volume[i] > 2.0 * vol_ma_20_1d_aligned[i]
        price_above_ema = close[i] > ema_50_1w_aligned[i]
        price_below_ema = close[i] < ema_50_1w_aligned[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long when: price > Donchian upper AND price > 1w EMA50 AND volume spike
            if close[i] > donchian_upper[i] and price_above_ema and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short when: price < Donchian lower AND price < 1w EMA50 AND volume spike
            elif close[i] < donchian_lower[i] and price_below_ema and vol_spike:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit when price crosses Donchian midpoint OR volume drops
            if close[i] < donchian_mid[i] or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit when price crosses Donchian midpoint OR volume drops
            if close[i] > donchian_mid[i] or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals