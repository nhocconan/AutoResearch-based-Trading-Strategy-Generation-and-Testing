#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Donchian channel breakout with 1d EMA34 trend filter and volume confirmation
# Long when price breaks above 1d Donchian upper (20) AND 1d EMA34 > EMA55 AND volume > 1.8 * avg_volume(20)
# Short when price breaks below 1d Donchian lower (20) AND 1d EMA34 < EMA55 AND volume > 1.8 * avg_volume(20)
# Exit when price crosses 1d EMA34 (trend reversal signal)
# Uses discrete sizing 0.30 to balance return and drawdown control
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe
# 1d Donchian provides strong trend-following structure with clear breakout levels
# 1d EMA34/EMA55 filter ensures alignment with intermediate daily trend
# Volume confirmation filters weak breakouts
# Works in bull (breakouts above upper channel in uptrend) and bear (breakdowns below lower channel in downtrend)

name = "4h_1dDonchian20_1dEMA34Trend_Volume_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for Donchian channels and EMA trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 55:  # Need sufficient data for EMA55
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Donchian channels (20-period)
    high_series_1d = pd.Series(high_1d)
    low_series_1d = pd.Series(low_1d)
    donchian_upper_20 = high_series_1d.rolling(window=20, min_periods=20).max().values
    donchian_lower_20 = low_series_1d.rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d EMA34 and EMA55 for trend filter
    close_series_1d = pd.Series(close_1d)
    ema_34_1d = close_series_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_55_1d = close_series_1d.ewm(span=55, adjust=False, min_periods=55).mean().values
    
    # Align 1d indicators to 4h timeframe (wait for completed 1d bar)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper_20)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower_20)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    ema_55_aligned = align_htf_to_ltf(prices, df_1d, ema_55_1d)
    
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
            # Long: price breaks above 1d Donchian upper with 1d EMA34 > EMA55 and volume confirmation
            if (close[i] > donchian_upper_aligned[i] and close[i-1] <= donchian_upper_aligned[i-1] and 
                ema_34_aligned[i] > ema_55_aligned[i] and volume_confirm[i]):
                signals[i] = 0.30
                position = 1
            # Short: price breaks below 1d Donchian lower with 1d EMA34 < EMA55 and volume confirmation
            elif (close[i] < donchian_lower_aligned[i] and close[i-1] >= donchian_lower_aligned[i-1] and 
                  ema_34_aligned[i] < ema_55_aligned[i] and volume_confirm[i]):
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: price crosses below 1d EMA34 (trend reversal)
            if close[i] < ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: price crosses above 1d EMA34 (trend reversal)
            if close[i] > ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals