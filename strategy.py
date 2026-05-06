#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 1w Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation
# Long when price breaks above 1w Donchian upper channel AND 1w EMA50 > EMA200 AND volume > 1.5 * avg_volume(20)
# Short when price breaks below 1w Donchian lower channel AND 1w EMA50 < EMA200 AND volume > 1.5 * avg_volume(20)
# Exit when price crosses 1w EMA50 (trend reversal signal)
# Uses discrete sizing 0.25 to balance return and drawdown
# Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe
# 1w Donchian provides clear structure with proven breakout edge
# 1w EMA50/EMA200 filter ensures alignment with higher timeframe trend (works in bull/bear)
# Volume confirmation filters weak breakouts (reduces false signals)
# Works in both bull and bear markets by following the 1w trend

name = "1d_1wDonchian20_1wEMA50Trend_Volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data ONCE before loop for Donchian channels and EMA
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:  # Need sufficient data for EMA50
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w Donchian(20) channels based on previous 20 1w bars
    high_series_1w = pd.Series(high_1w)
    low_series_1w = pd.Series(low_1w)
    donchian_high_1w = high_series_1w.rolling(window=20, min_periods=20).max().shift(1).values
    donchian_low_1w = low_series_1w.rolling(window=20, min_periods=20).min().shift(1).values
    
    # Calculate 1w EMA50 and EMA200 for trend filter
    close_series_1w = pd.Series(close_1w)
    ema_50_1w = close_series_1w.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200_1w = close_series_1w.ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Get 1d data ONCE before loop for volume confirmation
    # (volume is already in prices, but we need to ensure sufficient data)
    if len(prices) < 20:  # Need sufficient data for volume average
        return np.zeros(n)
    
    # Align 1w Donchian levels and EMA to 1d timeframe (wait for completed 1w bar)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high_1w)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low_1w)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Calculate volume confirmation: volume > 1.5 * 20-period average volume
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(ema_200_1w_aligned[i]) or 
            np.isnan(avg_volume_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 1w Donchian upper channel with 1w EMA50 > EMA200 and volume confirmation
            if (close[i] > donchian_high_aligned[i] and close[i-1] <= donchian_high_aligned[i-1] and 
                ema_50_1w_aligned[i] > ema_200_1w_aligned[i] and volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 1w Donchian lower channel with 1w EMA50 < EMA200 and volume confirmation
            elif (close[i] < donchian_low_aligned[i] and close[i-1] >= donchian_low_aligned[i-1] and 
                  ema_50_1w_aligned[i] < ema_200_1w_aligned[i] and volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below 1w EMA50 (trend reversal)
            if close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above 1w EMA50 (trend reversal)
            if close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals