#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation.
# Long when price breaks above upper Donchian channel, price > 1w EMA50, and volume > 1.5x 20-bar avg.
# Short when price breaks below lower Donchian channel, price < 1w EMA50, and volume > 1.5x 20-bar avg.
# Exit when price reverts to the 20-day EMA (mean reversion to intermediate trend).
# Uses 1w EMA50 for higher timeframe trend alignment, targeting 15-25 trades/year on 1d.
# Trend filter avoids counter-trend trades in bear markets, volume confirmation reduces false signals.
# Works in bull markets via breakouts and in bear markets via short breakdowns with trend alignment.

name = "1d_Donchian20_1wEMA50_Trend_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Donchian channels (20-period) from daily data
    # Upper band: highest high over past 20 days
    # Lower band: lowest low over past 20 days
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Calculate 20-day EMA for exit (mean reversion target)
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for Donchian and EMA50
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or np.isnan(ema_20[i]) or 
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_ema_50_1w = ema_50_1w_aligned[i]
        curr_upper = donchian_upper[i]
        curr_lower = donchian_lower[i]
        curr_ema_20 = ema_20[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above upper Donchian, price > 1w EMA50, volume spike
            if (curr_close > curr_upper and 
                curr_close > curr_ema_50_1w and 
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian, price < 1w EMA50, volume spike
            elif (curr_close < curr_lower and 
                  curr_close < curr_ema_50_1w and 
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position
            # Exit condition: price reverts to 20-day EMA (mean reversion)
            if curr_close <= curr_ema_20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit condition: price reverts to 20-day EMA (mean reversion)
            if curr_close >= curr_ema_20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals