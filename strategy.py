#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 1w Donchian channel breakout with 1d EMA34 trend filter and volume confirmation
# Long when price breaks above 1w Donchian upper (20) AND 1d EMA34 > EMA55 AND volume > 1.8 * avg_volume(20)
# Short when price breaks below 1w Donchian lower (20) AND 1d EMA34 < EMA55 AND volume > 1.8 * avg_volume(20)
# Exit when price crosses 1d EMA34 (trend reversal signal)
# Uses discrete sizing 0.25 to balance return and drawdown control
# Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe
# 1w Donchian provides strong trend-following structure with clear breakout levels
# 1d EMA34/EMA55 filter ensures alignment with intermediate daily trend
# Volume confirmation filters weak breakouts
# Works in bull (breakouts above upper channel in uptrend) and bear (breakdowns below lower channel in downtrend)

name = "1d_1wDonchian20_1dEMA34Trend_Volume"
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
    
    # Get 1w data ONCE before loop for Donchian channels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:  # Need sufficient data for Donchian
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate 1w Donchian channels (20-period)
    high_series_1w = pd.Series(high_1w)
    low_series_1w = pd.Series(low_1w)
    donchian_upper_20 = high_series_1w.rolling(window=20, min_periods=20).max().values
    donchian_lower_20 = low_series_1w.rolling(window=20, min_periods=20).min().values
    
    # Align 1w indicators to 1d timeframe (wait for completed 1w bar)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1w, donchian_upper_20)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1w, donchian_lower_20)
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 55:  # Need sufficient data for EMA55
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34 and EMA55 for trend filter
    close_series_1d = pd.Series(close_1d)
    ema_34_1d = close_series_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_55_1d = close_series_1d.ewm(span=55, adjust=False, min_periods=55).mean().values
    
    # Align 1d indicators to 1d timeframe (no delay needed as same timeframe)
    ema_34_aligned = ema_34_1d  # Same timeframe, no alignment needed
    ema_55_aligned = ema_55_1d  # Same timeframe, no alignment needed
    
    # Calculate volume confirmation: volume > 1.8 * 20-period average volume
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.8 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(ema_55_aligned[i]) or np.isnan(avg_volume_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 1w Donchian upper with 1d EMA34 > EMA55 and volume confirmation
            if (close[i] > donchian_upper_aligned[i] and close[i-1] <= donchian_upper_aligned[i-1] and 
                ema_34_aligned[i] > ema_55_aligned[i] and volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 1w Donchian lower with 1d EMA34 < EMA55 and volume confirmation
            elif (close[i] < donchian_lower_aligned[i] and close[i-1] >= donchian_lower_aligned[i-1] and 
                  ema_34_aligned[i] < ema_55_aligned[i] and volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below 1d EMA34 (trend reversal)
            if close[i] < ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above 1d EMA34 (trend reversal)
            if close[i] > ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals