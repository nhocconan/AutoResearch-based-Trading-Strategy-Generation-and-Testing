#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation
# Long when price breaks above upper Donchian(20) AND close > EMA50(1w) AND volume > 1.5x 20-period average
# Short when price breaks below lower Donchian(20) AND close < EMA50(1w) AND volume > 1.5x 20-period average
# Exit when price crosses back to the opposite Donchian level OR EMA50(1w) trend flips
# Uses 1d primary timeframe with 1w HTF for trend filter to reduce whipsaw
# Discrete sizing (0.25) to limit fee drag and manage drawdown
# Target: 30-100 trades over 4 years (7-25/year) to avoid fee drag

name = "1d_Donchian20_1wEMA50_Trend_Volume"
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
    
    # Get 1w data ONCE before loop for EMA50 trend filter and Donchian levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate EMA50 on 1w close for trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 1w Donchian channels from prior bar (to avoid look-ahead)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Prior bar's values (shifted by 1 to avoid look-ahead)
    prev_high = np.roll(high_1w, 1)
    prev_low = np.roll(low_1w, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    
    # Donchian(20): upper = max(high, lookback=20), lower = min(low, lookback=20)
    # Calculate on prior bar data
    upper_20 = np.full(len(prev_high), np.nan)
    lower_20 = np.full(len(prev_low), np.nan)
    
    for i in range(20, len(prev_high)):
        if not np.isnan(prev_high[i-20:i]).any() and not np.isnan(prev_low[i-20:i]).any():
            upper_20[i] = np.max(prev_high[i-20:i])
            lower_20[i] = np.min(prev_low[i-20:i])
    
    # Align 1w indicators to 1d timeframe
    upper_20_aligned = align_htf_to_ltf(prices, df_1w, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_1w, lower_20)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: volume > 1.5x 20-period average (balanced to avoid overtrading)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(upper_20_aligned[i]) or 
            np.isnan(lower_20_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above upper Donchian AND close > EMA50(1w) AND volume spike
            if (close[i] > upper_20_aligned[i] and 
                close[i] > ema_50_1w_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below lower Donchian AND close < EMA50(1w) AND volume spike
            elif (close[i] < lower_20_aligned[i] and 
                  close[i] < ema_50_1w_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below lower Donchian OR close < EMA50(1w) (trend flip)
            if (close[i] < lower_20_aligned[i] or 
                close[i] < ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above upper Donchian OR close > EMA50(1w) (trend flip)
            if (close[i] > upper_20_aligned[i] or 
                close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals