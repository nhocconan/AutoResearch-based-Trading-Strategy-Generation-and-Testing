#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation
# Long when price breaks above 1d Donchian high(20) AND 1w EMA50 > EMA200 AND volume > 2.0 * avg_volume(20)
# Short when price breaks below 1d Donchian low(20) AND 1w EMA50 < EMA200 AND volume > 2.0 * avg_volume(20)
# Exit when price crosses 1d EMA50 (trend reversal signal)
# Uses discrete sizing 0.25 to balance return and drawdown
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe
# 1d Donchian provides clear structure with proven breakout edge
# 1w EMA50/EMA200 filter ensures alignment with higher timeframe trend (works in bull/bear)
# Volume confirmation filters weak breakouts (reduces false signals)
# ATR-based stoploss to manage risk

name = "4h_1dDonchian20_1wEMA50Trend_Volume_v1"
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
    if len(df_1d) < 50:  # Need sufficient data for EMA50
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Donchian channels (20-period)
    high_series_1d = pd.Series(high_1d)
    low_series_1d = pd.Series(low_1d)
    donchian_high_20 = high_series_1d.rolling(window=20, min_periods=20).max().values
    donchian_low_20 = low_series_1d.rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d EMA50 for trend filter
    close_series_1d = pd.Series(close_1d)
    ema_50_1d = close_series_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200_1d = close_series_1d.ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Get 1w data ONCE before loop for EMA trend filter (higher timeframe)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:  # Need sufficient data for EMA50
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA50 and EMA200 for trend filter
    close_series_1w = pd.Series(close_1w)
    ema_50_1w = close_series_1w.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200_1w = close_series_1w.ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align 1d Donchian levels and EMA to 4h timeframe (wait for completed 1d bar)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high_20)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low_20)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Align 1w EMA indicators to 4h timeframe (wait for completed 1w bar)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Calculate volume confirmation: volume > 2.0 * 20-period average volume
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(ema_200_1d_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(ema_200_1w_aligned[i]) or 
            np.isnan(avg_volume_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 1d Donchian high with 1w EMA50 > EMA200 and volume confirmation
            if (close[i] > donchian_high_aligned[i] and close[i-1] <= donchian_high_aligned[i-1] and 
                ema_50_1w_aligned[i] > ema_200_1w_aligned[i] and volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 1d Donchian low with 1w EMA50 < EMA200 and volume confirmation
            elif (close[i] < donchian_low_aligned[i] and close[i-1] >= donchian_low_aligned[i-1] and 
                  ema_50_1w_aligned[i] < ema_200_1w_aligned[i] and volume_confirm[i]):
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