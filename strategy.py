#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation
# Long when price breaks above 4h Donchian upper band AND 12h EMA50 > 12h EMA200 AND volume > 1.5x 20-period average
# Short when price breaks below 4h Donchian lower band AND 12h EMA50 < 12h EMA200 AND volume > 1.5x 20-period average
# Exit when price crosses 4h Donchian middle band (mean reversion)
# Uses 4h primary timeframe with 12h HTF for trend filter (proven BTC/ETH edge)
# Volume confirmation ensures breakouts have conviction
# Discrete sizing (0.25) to limit fee drag and manage drawdown
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe

name = "4h_Donchian20_Breakout_12hEMA_Trend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data ONCE before loop for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA50 and EMA200 for trend filter
    def calculate_ema(data, span):
        return pd.Series(data).ewm(span=span, adjust=False, min_periods=span).mean().values
    
    ema_50_12h = calculate_ema(close_12h, 50)
    ema_200_12h = calculate_ema(close_12h, 200)
    
    # Align EMAs to 4h timeframe
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    ema_200_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_200_12h)
    
    # Trend filter: bullish when EMA50 > EMA200, bearish when EMA50 < EMA200
    trend_filter_bull = ema_50_12h_aligned > ema_200_12h_aligned
    trend_filter_bear = ema_50_12h_aligned < ema_200_12h_aligned
    
    # Calculate 4h Donchian channels (20-period)
    if len(high) >= 20:
        high_ma_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
        low_ma_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
        donchian_upper = high_ma_20
        donchian_lower = low_ma_20
        donchian_middle = (donchian_upper + donchian_lower) / 2
    else:
        donchian_upper = np.full(n, np.nan)
        donchian_lower = np.full(n, np.nan)
        donchian_middle = np.full(n, np.nan)
    
    # Volume confirmation: volume > 1.5x 20-period average
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(ema_200_12h_aligned[i]) or 
            np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or 
            np.isnan(donchian_middle[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian upper AND bullish trend AND volume spike
            if (close[i] > donchian_upper[i] and 
                trend_filter_bull[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian lower AND bearish trend AND volume spike
            elif (close[i] < donchian_lower[i] and 
                  trend_filter_bear[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below Donchian middle (mean reversion)
            if close[i] < donchian_middle[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above Donchian middle (mean reversion)
            if close[i] > donchian_middle[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals