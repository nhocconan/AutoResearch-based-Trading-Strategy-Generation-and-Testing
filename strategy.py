#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 1w Donchian(20) breakout with 1d EMA trend filter and volume confirmation
# Long when price breaks above 1w Donchian high(20) AND 1d EMA50 > EMA200 AND volume > 1.5 * avg_volume(20)
# Short when price breaks below 1w Donchian low(20) AND 1d EMA50 < EMA200 AND volume > 1.5 * avg_volume(20)
# Exit when price touches 1w Donchian midpoint or opposite Donchian band
# Uses discrete sizing 0.25 to balance return and drawdown control
# Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe
# 1w Donchian provides strong weekly structure for breakouts in both bull and bear markets
# 1d EMA50/EMA200 filter ensures alignment with daily trend, reducing counter-trend trades
# Volume confirmation (1.5x) filters weak breakouts while maintaining sufficient trade frequency

name = "1d_1wDonchian20_1dEMATrend_Volume"
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
    
    # Get 1w data ONCE before loop for Donchian channels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:  # Need sufficient data for Donchian(20)
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w Donchian(20) channels
    high_series_1w = pd.Series(high_1w)
    low_series_1w = pd.Series(low_1w)
    donchian_high_1w = high_series_1w.rolling(window=20, min_periods=20).max().values
    donchian_low_1w = low_series_1w.rolling(window=20, min_periods=20).min().values
    donchian_mid_1w = (donchian_high_1w + donchian_low_1w) / 2.0
    
    # Align 1w Donchian levels to 1d timeframe (wait for completed 1w bar)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high_1w)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low_1w)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_1w, donchian_mid_1w)
    
    # Calculate 1d EMA50 and EMA200 for trend filter
    close_series = pd.Series(close)
    ema_50 = close_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200 = close_series.ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Calculate volume confirmation: volume > 1.5 * 20-period average volume
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or np.isnan(ema_50[i]) or 
            np.isnan(ema_200[i]) or np.isnan(avg_volume_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 1w Donchian high with 1d EMA50 > EMA200 and volume confirmation
            if (close[i] > donchian_high_aligned[i] and close[i-1] <= donchian_high_aligned[i-1] and 
                ema_50[i] > ema_200[i] and volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 1w Donchian low with 1d EMA50 < EMA200 and volume confirmation
            elif (close[i] < donchian_low_aligned[i] and close[i-1] >= donchian_low_aligned[i-1] and 
                  ema_50[i] < ema_200[i] and volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price touches 1w Donchian midpoint or low band (profit take or reversal)
            if close[i] <= donchian_mid_aligned[i] or close[i] <= donchian_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price touches 1w Donchian midpoint or high band (profit take or reversal)
            if close[i] >= donchian_mid_aligned[i] or close[i] >= donchian_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals