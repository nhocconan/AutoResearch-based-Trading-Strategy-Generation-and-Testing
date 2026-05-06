#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Donchian channel breakout with 1d trend filter and volume confirmation
# Long when price breaks above 4h Donchian upper (20) AND 1d close > 1d EMA50 AND volume > 1.5 * avg_volume(20)
# Short when price breaks below 4h Donchian lower (20) AND 1d close < 1d EMA50 AND volume > 1.5 * avg_volume(20)
# Exit when price crosses 4h EMA34 (trend reversal signal)
# Uses discrete sizing 0.20 to minimize fee churn and control drawdown
# Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe
# 4h Donchian provides clear structure with proven breakout edge
# 1d EMA50 filter ensures alignment with higher timeframe trend (works in bull/bear)
# Volume confirmation filters weak breakouts (reduces false signals)
# Session filter (08-20 UTC) reduces noise trades during low-liquidity periods

name = "1h_4hDonchian20_1dEMA50_Trend_Volume_Session_v1"
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
    
    # Get 4h data ONCE before loop for Donchian channels and EMA
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 34:  # Need sufficient data for EMA34
        return np.zeros(n)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate 4h Donchian channels (20-period) based on previous 4h bar
    high_series_4h = pd.Series(high_4h)
    low_series_4h = pd.Series(low_4h)
    donchian_upper_4h = high_series_4h.rolling(window=20, min_periods=20).max().shift(1).values
    donchian_lower_4h = low_series_4h.rolling(window=20, min_periods=20).min().shift(1).values
    
    # Calculate 4h EMA34 for trend exit
    close_series_4h = pd.Series(close_4h)
    ema_34_4h = close_series_4h.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Get 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need sufficient data for EMA50
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    close_series_1d = pd.Series(close_1d)
    ema_50_1d = close_series_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 4h indicators to 1h timeframe (wait for completed 4h bar)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper_4h)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower_4h)
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Align 1d EMA50 to 1h timeframe (wait for completed 1d bar)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate volume confirmation: volume > 1.5 * 20-period average volume
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume_20)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = pd.DatetimeIndex(open_time).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(ema_34_4h_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(avg_volume_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Apply session filter
        if not session_filter[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 4h Donchian upper with 1d close > EMA50 and volume confirmation
            if (close[i] > donchian_upper_aligned[i] and close[i-1] <= donchian_upper_aligned[i-1] and 
                close[i] > ema_50_1d_aligned[i] and volume_confirm[i]):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below 4h Donchian lower with 1d close < EMA50 and volume confirmation
            elif (close[i] < donchian_lower_aligned[i] and close[i-1] >= donchian_lower_aligned[i-1] and 
                  close[i] < ema_50_1d_aligned[i] and volume_confirm[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price crosses below 4h EMA34 (trend reversal)
            if close[i] < ema_34_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price crosses above 4h EMA34 (trend reversal)
            if close[i] > ema_34_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals