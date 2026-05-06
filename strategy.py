#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Donchian channel breakout with 1h volume spike and choppiness regime filter
# Long when price breaks above 1d Donchian(20) high AND 1h volume > 2.0 * avg_volume(20) AND chop > 61.8 (range regime)
# Short when price breaks below 1d Donchian(20) low AND 1h volume > 2.0 * avg_volume(20) AND chop > 61.8 (range regime)
# Exit when price crosses 1d EMA50 (trend reversal)
# Uses discrete sizing 0.25 to balance return and drawdown
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe
# 1d Donchian provides clear structure with proven breakout edge in ranging markets
# Volume confirmation filters weak breakouts
# Choppiness regime ensures we only trade in ranging markets where mean reversion works

name = "4h_1dDonchian20_1hVolumeSpike_ChopRange_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for Donchian channels and EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need sufficient data for Donchian and EMA
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Donchian(20) channels based on previous 1d bar
    high_series_1d = pd.Series(high_1d)
    low_series_1d = pd.Series(low_1d)
    donchian_high_1d = high_series_1d.rolling(window=20, min_periods=20).max().shift(1).values
    donchian_low_1d = low_series_1d.rolling(window=20, min_periods=20).min().shift(1).values
    
    # Calculate 1d EMA50 for trend exit
    close_series_1d = pd.Series(close_1d)
    ema_50_1d = close_series_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Get 1h data ONCE before loop for volume and choppiness
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 30:  # Need sufficient data for volume and chop
        return np.zeros(n)
    volume_1h = df_1h['volume'].values
    high_1h = df_1h['high'].values
    low_1h = df_1h['low'].values
    close_1h = df_1h['close'].values
    
    # Calculate 1h volume confirmation: volume > 2.0 * 20-period average volume
    volume_series_1h = pd.Series(volume_1h)
    avg_volume_20_1h = volume_series_1h.rolling(window=20, min_periods=20).mean().values
    volume_spike_1h = volume_1h > (2.0 * avg_volume_20_1h)
    
    # Calculate 1h choppiness index: CHOP(14) = 100 * log10(sum(ATR(14)) / log10(range(14)))
    # Simplified: CHOP > 61.8 = ranging market (good for mean reversion/breakout fade)
    tr1 = np.maximum(high_1h[1:] - low_1h[:-1], np.abs(high_1h[1:] - close_1h[:-1]))
    tr1 = np.maximum(tr1, np.abs(low_1h[1:] - close_1h[:-1]))
    tr1 = np.concatenate([[np.nan], tr1])  # align with index
    
    atr_14 = pd.Series(tr1).rolling(window=14, min_periods=14).mean().values
    high_max_14 = pd.Series(high_1h).rolling(window=14, min_periods=14).max().values
    low_min_14 = pd.Series(low_1h).rolling(window=14, min_periods=14).min().values
    range_14 = high_max_14 - low_min_14
    
    # Avoid division by zero
    chop_1h = np.where(range_14 > 0, 100 * np.log10(atr_14 * 14 / range_14) / np.log10(14), 50)
    chop_1h = np.where(np.isnan(chop_1h), 50, chop_1h)
    chop_range = chop_1h > 61.8  # ranging market
    
    # Align 1d Donchian levels and EMA to 4h timeframe (wait for completed 1d bar)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high_1d)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low_1d)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Align 1h volume spike and chop to 4h timeframe (wait for completed 1h bar)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1h, volume_spike_1h)
    chop_range_aligned = align_htf_to_ltf(prices, df_1h, chop_range)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_spike_aligned[i]) or 
            np.isnan(chop_range_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 1d Donchian high with volume spike and chop > 61.8 (range)
            if (close[i] > donchian_high_aligned[i] and close[i-1] <= donchian_high_aligned[i-1] and 
                volume_spike_aligned[i] and chop_range_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 1d Donchian low with volume spike and chop > 61.8 (range)
            elif (close[i] < donchian_low_aligned[i] and close[i-1] >= donchian_low_aligned[i-1] and 
                  volume_spike_aligned[i] and chop_range_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below 1d EMA50 (trend reversal)
            if close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above 1d EMA50 (trend reversal)
            if close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals