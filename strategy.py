#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Donchian breakout with 1d EMA50 trend filter and volume confirmation
# Long when price breaks above 4h Donchian upper(20) AND 1d EMA50 > EMA200 AND volume > 1.5 * avg_volume(20)
# Short when price breaks below 4h Donchian lower(20) AND 1d EMA50 < EMA200 AND volume > 1.5 * avg_volume(20)
# Exit when price touches 4h Donchian middle (median of upper/lower) or opposite Donchian band
# Uses discrete sizing 0.20 to balance return and drawdown control
# Target: 60-150 total trades over 4 years = 15-37/year for 1h timeframe
# 4h Donchian provides strong structural support/resistance levels
# 1d EMA50/EMA200 filter ensures alignment with daily trend, reducing counter-trend trades
# Moderate volume confirmation (1.5x) filters weak breakouts while allowing sufficient trades
# Session filter (08-20 UTC) reduces noise trades during low-liquidity periods
# Works in bull (trend continuation breakouts above upper band) and bear (trend continuation breakdowns below lower band)

name = "1h_4hDonchian20_1dEMA50Trend_Volume_Session"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data ONCE before loop for Donchian channels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:  # Need sufficient data for Donchian(20)
        return np.zeros(n)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate 4h Donchian channels (20-period)
    high_series_4h = pd.Series(high_4h)
    low_series_4h = pd.Series(low_4h)
    upper_4h = high_series_4h.rolling(window=20, min_periods=20).max().values
    lower_4h = low_series_4h.rolling(window=20, min_periods=20).min().values
    middle_4h = (upper_4h + lower_4h) / 2.0
    
    # Align 4h Donchian levels to 1h timeframe (wait for completed 4h bar)
    upper_aligned = align_htf_to_ltf(prices, df_4h, upper_4h)
    lower_aligned = align_htf_to_ltf(prices, df_4h, lower_4h)
    middle_aligned = align_htf_to_ltf(prices, df_4h, middle_4h)
    
    # Get 1d data ONCE before loop for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:  # Need sufficient data for EMA200
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 and EMA200 for trend filter
    close_series_1d = pd.Series(close_1d)
    ema_50_1d = close_series_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200_1d = close_series_1d.ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align 1d EMA values to 1h timeframe (wait for completed 1d bar)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Calculate volume confirmation: volume > 1.5 * 20-period average volume
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):  # Start after warmup period
        # Skip if not in trading session or any value is NaN
        if not in_session[i] or (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or np.isnan(middle_aligned[i]) or 
                                 np.isnan(ema_50_aligned[i]) or np.isnan(ema_200_aligned[i]) or np.isnan(avg_volume_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 4h Donchian upper with 1d EMA50 > EMA200 and volume confirmation
            if (close[i] > upper_aligned[i] and close[i-1] <= upper_aligned[i-1] and 
                ema_50_aligned[i] > ema_200_aligned[i] and volume_confirm[i]):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below 4h Donchian lower with 1d EMA50 < EMA200 and volume confirmation
            elif (close[i] < lower_aligned[i] and close[i-1] >= lower_aligned[i-1] and 
                  ema_50_aligned[i] < ema_200_aligned[i] and volume_confirm[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price touches 4h Donchian middle or lower band (profit take or reversal)
            if close[i] <= middle_aligned[i] or close[i] <= lower_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price touches 4h Donchian middle or upper band (profit take or reversal)
            if close[i] >= middle_aligned[i] or close[i] >= upper_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals