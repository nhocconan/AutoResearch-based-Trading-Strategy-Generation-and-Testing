#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly EMA50 for trend filter
    close_1w_series = pd.Series(df_1w['close'].values)
    ema50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align weekly EMA50 to daily timeframe
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Get daily data for Donchian channels and volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Daily Donchian Channel (20)
    high_20 = pd.Series(df_1d['high'].values).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(df_1d['low'].values).rolling(window=20, min_periods=20).min().values
    
    # Align to daily timeframe (no additional alignment needed as we're already on 1d)
    high_20_aligned = high_20
    low_20_aligned = low_20
    
    # Daily volume filter: above average volume (20-period)
    vol_ma = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = vol_ma  # Already aligned to daily
    
    # Get minute-level data for entry timing (using 15m for better precision)
    df_15m = get_htf_data(prices, '15m')
    if len(df_15m) < 20:
        return np.zeros(n)
    
    # 15-minute volume spike detection
    vol_15m = pd.Series(df_15m['volume'].values)
    vol_ma_15m = vol_15m.rolling(window=20, min_periods=20).mean().values
    vol_spike = vol_15m > (vol_ma_15m * 1.5)  # 50% above average
    vol_spike_aligned = align_htf_to_ltf(prices, df_15m, vol_spike)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(high_20_aligned[i]) or np.isnan(low_20_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_ma_aligned[i]) or
            np.isnan(vol_spike_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below weekly EMA50
        trend_up = close[i] > ema50_1w_aligned[i]
        trend_down = close[i] < ema50_1w_aligned[i]
        
        # Volume filter: average volume + volume spike
        vol_filter = (volume[i] > vol_ma_aligned[i]) and vol_spike_aligned[i]
        
        # Entry conditions: 
        # Long: breakout above daily Donchian high in uptrend with volume spike
        # Short: breakdown below daily Donchian low in downtrend with volume spike
        long_breakout = close[i] > high_20_aligned[i]
        short_breakout = close[i] < low_20_aligned[i]
        
        long_entry = long_breakout and vol_filter and trend_up
        short_entry = short_breakout and vol_filter and trend_down
        
        # Exit conditions: opposite Donchian level touch
        long_exit = (close[i] < low_20_aligned[i]) and position == 1
        short_exit = (close[i] > high_20_aligned[i]) and position == -1
        
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

name = "1d_WeeklyEMA50_Trend_DailyDonchian_VolumeSpike"
timeframe = "1d"
leverage = 1.0