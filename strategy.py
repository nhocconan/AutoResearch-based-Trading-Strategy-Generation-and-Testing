#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA34 trend filter and volume spike confirmation
# Donchian(20) breakouts capture strong momentum moves with clear structure.
# 1w EMA34 provides higher-timeframe trend bias to avoid counter-trend trades.
# Volume spike (>2.0 x 20-period EMA) ensures institutional participation.
# Designed for 1d timeframe targeting 30-100 total trades over 4 years (7-25/year).
# Works in bull markets via trend-aligned breakouts and in bear markets via filtered mean-reversion at extreme levels.

name = "1d_Donchian20_1wEMA34_VolumeSpike_Trend"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 1w EMA34 for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate 1d Donchian channels (20-period)
    # Use rolling window on 1d data directly
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: 20-period EMA of volume
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike confirmation: current volume > 2.0 x 20-period EMA
        volume_spike = volume[i] > (2.0 * vol_ema_20[i])
        
        # 1w trend: bullish if close > EMA34, bearish if close < EMA34
        bullish_trend = close[i] > ema_34_1w_aligned[i]
        bearish_trend = close[i] < ema_34_1w_aligned[i]
        
        if position == 0:
            # Long: Close breaks above Donchian high + volume spike + bullish 1w trend
            if (close[i] > donchian_high[i] and volume_spike and bullish_trend):
                signals[i] = 0.25
                position = 1
            # Short: Close breaks below Donchian low + volume spike + bearish 1w trend
            elif (close[i] < donchian_low[i] and volume_spike and bearish_trend):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Close drops below Donchian low OR 1w trend turns bearish
            if (close[i] < donchian_low[i] or bearish_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Close rises above Donchian high OR 1w trend turns bullish
            if (close[i] > donchian_high[i] or bullish_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals