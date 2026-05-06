#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d EMA34 trend filter and volume spike
# Long when price breaks above 12h Donchian upper band (20-bar high) AND 1d EMA34 uptrend AND volume > 2.0 * 20-bar avg volume
# Short when price breaks below 12h Donchian lower band (20-bar low) AND 1d EMA34 downtrend AND volume > 2.0 * 20-bar avg volume
# Exit with signal=0 when price reverses back inside the 12h Donchian H-L range (mean reversion)
# Uses discrete sizing 0.25 to balance opportunity and drawdown
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe
# Donchian channels provide clear structure; breakouts capture momentum with filters to avoid false signals
# 1d EMA34 ensures higher-timeframe trend alignment to avoid counter-trend trades
# Volume spike confirms institutional participation
# Works in bull via buying strength on upside breakouts, works in bear via selling strength on downside breakdowns

name = "12h_Donchian20_1dEMA34_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data ONCE before loop for Donchian calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h Donchian channels (20-period)
    high_20 = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Align HTF indicators to 12h timeframe (wait for completed HTF bar)
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, high_20)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, low_20)
    
    # Get 1d data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34
    close_1d_series = pd.Series(close_1d)
    ema_34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align HTF indicators to 12h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate volume confirmation: volume > 2.0 * 20-bar average volume (stricter for fewer trades)
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Donchian breakout signals with trend and volume filters
            # Long: price breaks above Donchian upper band AND uptrend AND volume spike
            if close[i] > donchian_high_aligned[i] and close[i] > ema_34_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower band AND downtrend AND volume spike
            elif close[i] < donchian_low_aligned[i] and close[i] < ema_34_1d_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price reverses back inside Donchian H-L range (mean reversion)
            if close[i] < donchian_high_aligned[i] and close[i] > donchian_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price reverses back inside Donchian H-L range (mean reversion)
            if close[i] < donchian_high_aligned[i] and close[i] > donchian_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals