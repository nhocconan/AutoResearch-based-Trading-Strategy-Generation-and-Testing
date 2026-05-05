#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation
# Long when price breaks above upper Donchian(20) AND close > EMA50(1w) AND volume > 2.0x 20-period average
# Short when price breaks below lower Donchian(20) AND close < EMA50(1w) AND volume > 2.0x 20-period average
# Exit when price retracement to Donchian midpoint OR EMA50(1w) trend flip
# Uses 1d primary timeframe with 1w HTF for trend filter to reduce whipsaw and avoid overtrading
# Discrete sizing (0.25) to limit fee drag and manage drawdown
# Target: 30-100 total trades over 4 years (7-25/year) to avoid fee drag
# Donchian channels provide robust structure; breakouts with volume and trend filter capture strong moves

name = "1d_Donchian20_Breakout_1wEMA50_Trend_Volume"
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
    
    # Get 1w data ONCE before loop for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate EMA50 on 1w close for trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Get 1d data for Donchian channels (based on previous 20 periods to avoid look-ahead)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Donchian(20) from 1d OHLC (using previous 20 periods)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Upper band: max(high, lookback=20)
    upper_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    # Lower band: min(low, lookback=20)
    lower_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    # Midpoint: (upper + lower) / 2
    midpoint_20 = (upper_20 + lower_20) / 2.0
    
    # Align to 1d timeframe (Donchian levels are already aligned as we used 1d data)
    upper_aligned = upper_20
    lower_aligned = lower_20
    midpoint_aligned = midpoint_20
    
    # Volume confirmation: volume > 2.0x 20-period average (strict to reduce trades)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (2.0 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(upper_aligned[i]) or 
            np.isnan(lower_aligned[i]) or 
            np.isnan(midpoint_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above upper Donchian(20) AND close > EMA50(1w) AND volume spike
            if (high[i] > upper_aligned[i] and 
                close[i] > ema_50_1w_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below lower Donchian(20) AND close < EMA50(1w) AND volume spike
            elif (low[i] < lower_aligned[i] and 
                  close[i] < ema_50_1w_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price retracement to Donchian midpoint OR close < EMA50(1w) (trend flip)
            if close[i] <= midpoint_aligned[i] or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price retracement to Donchian midpoint OR close > EMA50(1w) (trend flip)
            if close[i] >= midpoint_aligned[i] or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals