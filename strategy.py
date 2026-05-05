#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation
# Long when price breaks above Donchian upper band AND close > EMA34(1d) AND volume > 1.8x 20-period average
# Short when price breaks below Donchian lower band AND close < EMA34(1d) AND volume > 1.8x 20-period average
# Exit when price retraces to Donchian midpoint OR EMA34(1d) trend flip
# Uses 4h primary timeframe with 1d HTF for trend filter to reduce whipsaw and avoid overtrading
# Discrete sizing (0.25) to limit fee drag and manage drawdown
# Target: 75-150 total trades over 4 years (19-37/year) to avoid fee drag
# Donchian channels provide clear structure; breakouts with volume and daily trend filter capture strong moves

name = "4h_Donchian20_Breakout_1dEMA34_Trend_Volume"
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
    
    # Get 1d data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate EMA34 on 1d close for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Donchian channels on 4h data (using previous 20 periods to avoid look-ahead)
    if len(high) >= 20:
        # Rolling max/min with min_periods=20 ensures we only use completed windows
        upper_band = pd.Series(high).rolling(window=20, min_periods=20).max().values
        lower_band = pd.Series(low).rolling(window=20, min_periods=20).min().values
        middle_band = (upper_band + lower_band) / 2.0
    else:
        upper_band = np.full(n, np.nan)
        lower_band = np.full(n, np.nan)
        middle_band = np.full(n, np.nan)
    
    # Volume confirmation: volume > 1.8x 20-period average (balanced to avoid overtrading)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.8 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(upper_band[i]) or 
            np.isnan(lower_band[i]) or 
            np.isnan(middle_band[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian upper band AND close > EMA34(1d) AND volume spike
            if (high[i] > upper_band[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian lower band AND close < EMA34(1d) AND volume spike
            elif (low[i] < lower_band[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price retraces to Donchian midpoint OR close < EMA34(1d) (trend flip)
            if close[i] <= middle_band[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price retraces to Donchian midpoint OR close > EMA34(1d) (trend flip)
            if close[i] >= middle_band[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals