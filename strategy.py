#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation
# Donchian channel breakouts from weekly highs/lows capture strong momentum moves.
# The 1w EMA50 acts as a smooth trend filter to avoid counter-trend trades.
# Volume confirmation (2.0x 50-period EMA) ensures breakouts have conviction.
# Designed for 1d timeframe to target 15-25 trades/year (60-100 total over 4 years)
# with discrete sizing (0.25). Works in bull markets by buying breakouts above
# weekly Donchian high in uptrends and in bear markets by selling breakdowns
# below weekly Donchian low in downtrends.

name = "1d_Donchian20_1wEMA50_Trend_VolumeSpike"
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
    
    # Get 1w data for Donchian levels and EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Donchian levels: 20-period high/low from 1w
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate rolling max/min for Donchian channels
    high_roll = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 1d timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, high_roll)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, low_roll)
    
    # Volume confirmation: 2.0x 50-period EMA on 1d volume
    vol_series = pd.Series(volume)
    vol_ema_50 = vol_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start from 100 to have valid indicators
        # Skip if any value is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or np.isnan(vol_ema_50[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 2.0 x 50-period EMA
        volume_confirmed = volume[i] > (2.0 * vol_ema_50[i])
        
        if position == 0:
            # Long: close breaks above Donchian high + volume confirmation + price above 1w EMA50 (uptrend)
            if (close[i] > donchian_high_aligned[i] and volume_confirmed and 
                close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: close breaks below Donchian low + volume confirmation + price below 1w EMA50 (downtrend)
            elif (close[i] < donchian_low_aligned[i] and volume_confirmed and 
                  close[i] < ema_50_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price falls below Donchian low (mean reversion) OR below 1w EMA50 (trend change)
            if close[i] < donchian_low_aligned[i] or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price rises above Donchian high (mean reversion) OR above 1w EMA50 (trend change)
            if close[i] > donchian_high_aligned[i] or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals