#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation.
# Long when price breaks above upper Donchian channel, close > 1w EMA50, and volume > 2.0x 20-bar avg.
# Short when price breaks below lower Donchian channel, close < 1w EMA50, and volume > 2.0x 20-bar avg.
# Exit when price re-enters the Donchian channel (between upper and lower bands).
# Uses 1d timeframe for optimal trade frequency (target: 7-25 trades/year) to minimize fee drag.
# Donchian channels provide clear breakout/breakdown levels based on 20-day price extremes.
# 1w EMA50 filters for higher timeframe trend alignment to avoid counter-trend trades.
# Volume confirmation with higher threshold reduces false breakouts.
# Works in bull markets via breakouts with trend and in bear markets via breakdowns with trend.
# Target: 30-100 total trades over 4 years.

name = "1d_Donchian20_1wEMA50_Trend_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Donchian channels from previous 20-period high/low
    # Use rolling window on 1d data to get 20-day high/low
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Donchian levels (upper and lower bands)
    upper_band = high_20
    lower_band = low_20
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # warmup for EMA50 and Donchian channels
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_upper = upper_band[i]
        curr_lower = lower_band[i]
        curr_ema_50_1w = ema_50_1w_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above upper band, close > 1w EMA50, volume spike
            if (curr_close > curr_upper and 
                curr_close > curr_ema_50_1w and 
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower band, close < 1w EMA50, volume spike
            elif (curr_close < curr_lower and 
                  curr_close < curr_ema_50_1w and 
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position
            # Exit condition: price re-enters the Donchian channel (below upper band)
            if curr_close < curr_upper:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit condition: price re-enters the Donchian channel (above lower band)
            if curr_close > curr_lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals