#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Donchian channel breakout with 1d EMA50 trend filter and volume confirmation
# Long when price breaks above 4h Donchian upper (20-period) AND 1d EMA50 > EMA200 AND volume > 2.0 * avg_volume(20)
# Short when price breaks below 4h Donchian lower (20-period) AND 1d EMA50 < EMA200 AND volume > 2.0 * avg_volume(20)
# Exit when price crosses 4h EMA50 (trend reversal signal)
# Uses discrete sizing 0.20 to minimize fee impact and control drawdown
# Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe
# 4h Donchian provides clear structure with proven breakout edge
# 1d EMA50/EMA200 filter ensures alignment with higher timeframe trend (works in bull/bear)
# Volume confirmation filters weak breakouts (reduces false signals)
# Session filter (08-20 UTC) reduces noise during low-liquidity periods
# Works in bull (breakouts above upper in uptrend) and bear (breakdowns below lower in downtrend)

name = "1h_4hDonchian20_1dEMA50Trend_Volume_v1"
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
    open_time = prices['open_time'].values
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data ONCE before loop for Donchian channels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:  # Need sufficient data for Donchian calculation
        return np.zeros(n)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate 4h Donchian channels (20-period) based on previous 4h bar
    donchian_upper_4h = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_lower_4h = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Get 1d data ONCE before loop for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:  # Need sufficient data for EMA200
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 and EMA200 for trend filter
    close_series_1d = pd.Series(close_1d)
    ema_50_1d = close_series_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200_1d = close_series_1d.ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align 4h Donchian levels to 1h timeframe (wait for completed 4h bar)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper_4h)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower_4h)
    
    # Align 1d EMA indicators to 1h timeframe (wait for completed 1d bar)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Calculate volume confirmation: volume > 2.0 * 20-period average volume
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(ema_200_aligned[i]) or np.isnan(avg_volume_20[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 4h Donchian upper with 1d EMA50 > EMA200 and volume confirmation
            if (close[i] > donchian_upper_aligned[i] and close[i-1] <= donchian_upper_aligned[i-1] and 
                ema_50_aligned[i] > ema_200_aligned[i] and volume_confirm[i]):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below 4h Donchian lower with 1d EMA50 < EMA200 and volume confirmation
            elif (close[i] < donchian_lower_aligned[i] and close[i-1] >= donchian_lower_aligned[i-1] and 
                  ema_50_aligned[i] < ema_200_aligned[i] and volume_confirm[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price crosses below 4h EMA50 (trend reversal)
            # Note: We use 4h EMA50 for exit to match the timeframe of entry signal
            df_4h_close = get_htf_data(prices, '4h')['close'].values
            ema_50_4h = pd.Series(df_4h_close).ewm(span=50, adjust=False, min_periods=50).mean().values
            ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
            if close[i] < ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price crosses above 4h EMA50 (trend reversal)
            df_4h_close = get_htf_data(prices, '4h')['close'].values
            ema_50_4h = pd.Series(df_4h_close).ewm(span=50, adjust=False, min_periods=50).mean().values
            ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
            if close[i] > ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals