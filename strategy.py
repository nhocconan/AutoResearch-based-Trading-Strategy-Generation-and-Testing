#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 12h Donchian Breakout with Volume Confirmation and Weekly Trend Filter
# Hypothesis: Donchian breakouts capture momentum; volume confirms institutional participation;
# weekly trend filter ensures alignment with higher timeframe bias, reducing false signals
# in both bull and bear markets. Designed for 12h timeframe with low trade frequency.

name = "12h_donchian_breakout_weekly_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 20:
        return np.zeros(n)
    
    close_weekly = df_weekly['close'].values
    
    # Weekly EMA(20) for trend filter
    ema_20_weekly = pd.Series(close_weekly).ewm(span=20, adjust=False).mean().values
    ema_20_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_20_weekly)
    
    # Daily data for Donchian channels and volume
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 20:
        return np.zeros(n)
    
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    close_daily = df_daily['close'].values
    volume_daily = df_daily['volume'].values
    
    # Donchian channels (20-day)
    donchian_high = pd.Series(high_daily).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_daily).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: daily volume > 1.5x 20-day average
    vol_ma_20d = pd.Series(volume_daily).rolling(window=20, min_periods=10).mean().values
    vol_spike = volume_daily > (1.5 * vol_ma_20d)
    
    # Align daily indicators to 12h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_daily, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_daily, donchian_low)
    vol_spike_aligned = align_htf_to_ltf(prices, df_daily, vol_spike)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema_20_weekly_aligned[i]) or np.isnan(vol_spike_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Check volume confirmation
        vol_ok = vol_spike_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price breaks below Donchian low or weekly trend turns bearish
            if close[i] <= donchian_low_aligned[i] or close[i] < ema_20_weekly_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high or weekly trend turns bullish
            if close[i] >= donchian_high_aligned[i] or close[i] > ema_20_weekly_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if vol_ok:
                # Long: price breaks above Donchian high in bullish weekly trend
                if close[i] > donchian_high_aligned[i] and close[i] > ema_20_weekly_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short: price breaks below Donchian low in bearish weekly trend
                elif close[i] < donchian_low_aligned[i] and close[i] < ema_20_weekly_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals