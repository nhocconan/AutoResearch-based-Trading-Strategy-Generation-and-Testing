#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA34 trend filter and volume confirmation
# Long when price breaks above 1d Donchian upper(20) AND 1w close > 1w EMA34 AND volume > 2.0x 50-period average
# Short when price breaks below 1d Donchian lower(20) AND 1w close < 1w EMA34 AND volume > 2.0x 50-period average
# Exit when price crosses 1d Donchian midpoint (mean of upper/lower)
# Uses 1d primary timeframe with 1w HTF for trend filter and Donchian structure
# Discrete sizing (0.30) to limit fee drag and manage drawdown
# Target: 30-100 total trades over 4 years (7-25/year) based on proven Donchian breakout performance
# Donchian channels provide robust price structure; 1w EMA34 filters for higher-timeframe trend; volume confirms breakout validity
# Works in both bull and bear markets by following the 1w trend while using 1d for entry timing

name = "1d_Donchian20_Breakout_1wEMA34_Trend_Volume"
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
    
    # Get 1w data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate EMA34 on 1w close for trend filter
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate Donchian(20) on 1d data (based on 20-period high/low)
    if len(high) >= 20:
        donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
        donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
        donchian_mid = (donchian_high + donchian_low) / 2.0
    else:
        donchian_high = np.full(n, np.nan)
        donchian_low = np.full(n, np.nan)
        donchian_mid = np.full(n, np.nan)
    
    # Volume confirmation: volume > 2.0x 50-period average
    if len(volume) >= 50:
        vol_ma_50 = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
        volume_filter = volume > (2.0 * vol_ma_50)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(donchian_mid[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian upper(20) AND 1w close > 1w EMA34 AND volume spike
            if (close[i] > donchian_high[i] and 
                close[i] > ema_34_1w_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.30
                position = 1
            # Short conditions: price breaks below Donchian lower(20) AND 1w close < 1w EMA34 AND volume spike
            elif (close[i] < donchian_low[i] and 
                  close[i] < ema_34_1w_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: price crosses below Donchian midpoint (trend reversal)
            if close[i] < donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: price crosses above Donchian midpoint (trend reversal)
            if close[i] > donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals