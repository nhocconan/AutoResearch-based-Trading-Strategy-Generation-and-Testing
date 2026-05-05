#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA34 trend filter and volume confirmation
# Long when price breaks above Donchian upper(20) AND close > EMA34(1w) AND volume > 2.0x 20-period average
# Short when price breaks below Donchian lower(20) AND close < EMA34(1w) AND volume > 2.0x 20-period average
# Exit when price retraces to Donchian midpoint OR close crosses EMA34(1w) (trend flip)
# Uses 1d primary timeframe with 1w HTF for trend filter to capture multi-week moves with controlled frequency
# Discrete sizing (0.25) to limit fee drag and manage drawdown in both bull and bear markets
# Target: 30-100 total trades over 4 years (7-25/year) to avoid fee drag
# Donchian channels provide strong structural breakouts; EMA34(1w) filter ensures alignment with weekly trend
# Volume confirmation avoids low-conviction breakouts, improving win rate in choppy markets

name = "1d_Donchian20_Breakout_1wEMA34_Trend_Volume"
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
    
    # Get 1w data ONCE before loop for EMA34 trend filter and Donchian levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate EMA34 on 1w close for trend filter
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate Donchian levels from 1w OHLC (using previous 1w bar's values to avoid look-ahead)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Donchian(20): upper = max(high,20), lower = min(low,20), midpoint = (upper+lower)/2
    upper_20_1w = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    lower_20_1w = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    midpoint_20_1w = (upper_20_1w + lower_20_1w) / 2.0
    
    # Align to 1d timeframe (using previous 1w bar's levels to avoid look-ahead)
    upper_aligned = align_htf_to_ltf(prices, df_1w, upper_20_1w)
    lower_aligned = align_htf_to_ltf(prices, df_1w, lower_20_1w)
    midpoint_aligned = align_htf_to_ltf(prices, df_1w, midpoint_20_1w)
    
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
            np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian upper(20) AND close > EMA34(1w) AND volume spike
            if (high[i] > upper_aligned[i] and 
                close[i] > ema_34_1w_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian lower(20) AND close < EMA34(1w) AND volume spike
            elif (low[i] < lower_aligned[i] and 
                  close[i] < ema_34_1w_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price retracement to Donchian midpoint OR close < EMA34(1w) (trend flip)
            if close[i] <= midpoint_aligned[i] or close[i] < ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price retracement to Donchian midpoint OR close > EMA34(1w) (trend flip)
            if close[i] >= midpoint_aligned[i] or close[i] > ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals