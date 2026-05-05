#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout + 1d EMA50 trend + volume confirmation
# Long when price breaks above 20-period Donchian upper band AND close > 1d EMA50 AND volume > 1.5x 20-period average
# Short when price breaks below 20-period Donchian lower band AND close < 1d EMA50 AND volume > 1.5x 20-period average
# Exit when price crosses the 20-period Donchian middle band (mean of upper/lower)
# Uses 12h primary timeframe with 1d HTF for trend filter to capture multi-day moves with controlled frequency
# Discrete sizing (0.25) to limit fee drag and manage drawdown in both bull and bear markets
# Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag
# Donchian channels provide clear structure; EMA50 filters for higher-timeframe trend; volume confirms participation

name = "12h_Donchian20_Breakout_1dEMA50_Trend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate EMA50 on 1d close for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Donchian channels on 12h data (20-period)
    if len(high) >= 20:
        donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
        donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
        donchian_middle = (donchian_upper + donchian_lower) / 2.0
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
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or np.isnan(donchian_middle[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: break above upper band AND close > 1d EMA50 AND volume spike
            if close[i] > donchian_upper[i] and close[i] > ema_50_1d_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short conditions: break below lower band AND close < 1d EMA50 AND volume spike
            elif close[i] < donchian_lower[i] and close[i] < ema_50_1d_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below middle band (trend weakening)
            if close[i] < donchian_middle[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above middle band (trend weakening)
            if close[i] > donchian_middle[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals