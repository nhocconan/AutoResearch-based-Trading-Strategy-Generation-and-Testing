#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation
# Long when price breaks above 1d Donchian upper channel AND 1w EMA50 > 1w EMA200 AND volume > 1.5x 20-period average
# Short when price breaks below 1d Donchian lower channel AND 1w EMA50 < 1w EMA200 AND volume > 1.5x 20-period average
# Exit when price crosses 1d Donchian midpoint (mean reversion) OR 1w EMA50/200 cross reverses
# Uses 1d primary timeframe with 1w HTF for EMA trend filter
# Donchian channels provide clear breakout zones based on recent price action
# EMA filter ensures we only trade in strong trending markets, reducing whipsaw
# Volume confirmation filters low-momentum breakouts
# Discrete sizing (0.25) to limit fee drag and manage drawdown
# Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe

name = "1d_Donchian20_Breakout_1wEMA_Trend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data ONCE before loop for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 and EMA200 for trend filter
    close_1w = df_1w['close'].values
    
    # EMA50
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    # EMA200
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align EMAs to 1d timeframe
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # Calculate 1d Donchian channels (20-period)
    if len(high) >= 20:
        # Upper channel: highest high of last 20 periods
        donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
        # Lower channel: lowest low of last 20 periods
        donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
        # Midpoint: average of upper and lower channels
        donchian_mid = (donchian_upper + donchian_lower) / 2
    else:
        donchian_upper = np.full(n, np.nan)
        donchian_lower = np.full(n, np.nan)
        donchian_mid = np.full(n, np.nan)
    
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
        if (np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(ema200_1w_aligned[i]) or 
            np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or 
            np.isnan(donchian_mid[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian upper AND EMA50 > EMA200 AND volume spike
            if (close[i] > donchian_upper[i] and 
                ema50_1w_aligned[i] > ema200_1w_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian lower AND EMA50 < EMA200 AND volume spike
            elif (close[i] < donchian_lower[i] and 
                  ema50_1w_aligned[i] < ema200_1w_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below Donchian midpoint (mean reversion) OR EMA50 < EMA200 (trend weakening)
            if close[i] < donchian_mid[i] or ema50_1w_aligned[i] < ema200_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above Donchian midpoint (mean reversion) OR EMA50 > EMA200 (trend weakening)
            if close[i] > donchian_mid[i] or ema50_1w_aligned[i] > ema200_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals