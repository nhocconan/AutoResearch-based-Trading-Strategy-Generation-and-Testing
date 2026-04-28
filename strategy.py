#!/usr/bin/env python3
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
    
    # Get 12h data for Donchian channels and trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Get 1d data for volume confirmation and regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 12h Donchian channels (20 periods)
    high_12h_series = pd.Series(df_12h['high'].values)
    low_12h_series = pd.Series(df_12h['low'].values)
    donchian_upper = high_12h_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_12h_series.rolling(window=20, min_periods=20).min().values
    donchian_middle = (donchian_upper + donchian_lower) / 2
    
    # Align Donchian channels to 6h timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_12h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_12h, donchian_lower)
    donchian_middle_aligned = align_htf_to_ltf(prices, df_12h, donchian_middle)
    
    # 12h EMA50 for trend filter
    close_12h_series = pd.Series(df_12h['close'].values)
    ema50_12h = close_12h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # 1d volume spike detection (volume > 1.5x 20-period average)
    volume_1d_series = pd.Series(df_1d['volume'].values)
    vol_ma_20 = volume_1d_series.rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_1d_series > (vol_ma_20 * 1.5)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike.values)
    
    # 6h volume filter: above average volume
    vol_ma_6h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Hour filter: 8-20 UTC (most active trading hours)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(ema50_12h_aligned[i]) or np.isnan(vol_spike_aligned[i]) or 
            np.isnan(vol_ma_6h[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 8-20 UTC
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            # Outside session: flatten position
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Volume filters: 6h volume above average AND 12h volume spike
        vol_filter_6h = volume[i] > vol_ma_6h[i]
        vol_filter_12h = vol_spike_aligned[i] == 1
        
        # Trend filter: price above/below 12h EMA50
        trend_up = close[i] > ema50_12h_aligned[i]
        trend_down = close[i] < ema50_12h_aligned[i]
        
        # Entry conditions: 
        # Long: price breaks above Donchian upper in uptrend with volume confirmation
        # Short: price breaks below Donchian lower in downtrend with volume confirmation
        long_entry = (close[i] > donchian_upper_aligned[i]) and vol_filter_6h and vol_filter_12h and trend_up
        short_entry = (close[i] < donchian_lower_aligned[i]) and vol_filter_6h and vol_filter_12h and trend_down
        
        # Exit conditions: price returns to Donchian middle
        long_exit = (close[i] < donchian_middle_aligned[i]) and position == 1
        short_exit = (close[i] > donchian_middle_aligned[i]) and position == -1
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit:
            signals[i] = 0.0
            position = 0
        elif short_exit:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_DonchianBreakout_12hTrend_VolumeSpike_Session"
timeframe = "6h"
leverage = 1.0