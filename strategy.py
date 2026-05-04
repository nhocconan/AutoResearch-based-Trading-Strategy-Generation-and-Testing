#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout + 1w EMA34 trend + volume confirmation
# Uses 1w EMA34 to establish primary trend (bullish if close > EMA34, bearish if close < EMA34)
# Enters long when price breaks above 1d Donchian upper channel + volume > 1.5 x 20-period EMA + bullish 1w trend
# Enters short when price breaks below 1d Donchian lower channel + volume > 1.5 x 20-period EMA + bearish 1w trend
# Exits on opposite Donchian breakout or when 1w trend reverses
# Volume spike confirms institutional participation, reducing false breakouts
# Designed for 1d timeframe targeting 7-25 trades/year with discrete sizing (0.25)
# Works in bull markets (breakouts with volume in uptrend) and bear markets (breakouts with volume in downtrend)

name = "1d_Donchian20_1wEMA34_VolumeSpike_Trend"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 1w EMA34
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Get 1d data for Donchian channels (20-period)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d Donchian channels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    upper_channel = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lower_channel = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    upper_aligned = align_htf_to_ltf(prices, df_1d, upper_channel)
    lower_aligned = align_htf_to_ltf(prices, df_1d, lower_channel)
    
    # Get 1d data for volume EMA(20) for volume confirmation
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d volume EMA(20) for volume confirmation
    vol_1d = df_1d['volume'].values
    vol_ema_20 = pd.Series(vol_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ema_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(upper_aligned[i]) or 
            np.isnan(lower_aligned[i]) or np.isnan(vol_ema_20_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current 1d volume > 1.5 x 20-period EMA
        volume_confirmed = volume[i] > (1.5 * vol_ema_20_aligned[i])
        
        # 1w trend: bullish if close > EMA34, bearish if close < EMA34
        bullish_trend = close[i] > ema_34_1w_aligned[i]
        bearish_trend = close[i] < ema_34_1w_aligned[i]
        
        if position == 0:
            # Long: price breaks above upper Donchian + volume confirmation + bullish 1w trend
            if (close[i] > upper_aligned[i] and volume_confirmed and bullish_trend):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian + volume confirmation + bearish 1w trend
            elif (close[i] < lower_aligned[i] and volume_confirmed and bearish_trend):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price falls below lower Donchian OR 1w trend turns bearish
            if close[i] < lower_aligned[i] or bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price rises above upper Donchian OR 1w trend turns bullish
            if close[i] > upper_aligned[i] or bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals