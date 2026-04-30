#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation.
# Long when price breaks above upper Donchian channel, 1d EMA50 uptrend, and volume > 1.5x 20-bar avg.
# Short when price breaks below lower Donchian channel, 1d EMA50 downtrend, and volume > 1.5x 20-bar avg.
# Exit on opposite Donchian channel touch (lower for long exit, upper for short exit).
# Donchian channels provide robust trend-following structure proven on SOLUSDT.
# Combined with 1d EMA50 trend filter and volume confirmation to reduce false signals.
# Timeframe: 4h as per experiment guidelines.

name = "4h_Donchian20_1dEMA50_Trend_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Donchian(20) channels from prior completed 20 periods
    # Upper = max(high over last 20 periods), Lower = min(low over last 20 periods)
    # Use rolling window with min_periods=20 to ensure we only use completed windows
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for EMA50 and Donchian calculation
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema_50_1d = ema_50_1d_aligned[i]
        curr_upper = donchian_upper[i]
        curr_lower = donchian_lower[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above upper Donchian, uptrend (close > 1d EMA50), volume spike
            if (curr_close > curr_upper and 
                curr_close > curr_ema_50_1d and 
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian, downtrend (close < 1d EMA50), volume spike
            elif (curr_close < curr_lower and 
                  curr_close < curr_ema_50_1d and 
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position
            # Exit condition: price touches or goes below lower Donchian (mean reversion)
            if curr_close <= curr_lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit condition: price touches or goes above upper Donchian (mean reversion)
            if curr_close >= curr_upper:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals