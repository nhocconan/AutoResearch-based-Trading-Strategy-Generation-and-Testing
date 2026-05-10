#/usr/bin/env python3
# 4h_Volume_Crush_Breakout_1dTrend_Volume
# Hypothesis: After a volume surge (vol > 2.0 * 20-period average), price often breaks out in the direction of the daily trend. This captures momentum after periods of accumulation/distribution. Daily trend filter ensures we trade with the higher timeframe momentum, reducing counter-trend trades. Volume surge acts as a catalyst for breakout, improving signal quality. Designed for low frequency (~20-50 trades/year) to minimize fee drag.

name = "4h_Volume_Crush_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Volume surge: volume > 2.0 * 20-period average
    volume_series = pd.Series(volume)
    vol_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_surge = volume > (2.0 * vol_ma)
    
    # Daily trend: EMA34 on daily close
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_1d_up = close_1d > ema34_1d
    trend_1d_down = close_1d < ema34_1d
    
    # Align daily trend to 4h
    trend_1d_up_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_up.astype(float))
    trend_1d_down_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_down.astype(float))
    
    # Donchian breakout (20-period) for entry
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(trend_1d_up_aligned[i]) or np.isnan(trend_1d_down_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: Donchian high breakout + volume surge + daily uptrend
            if (close[i] > donchian_high[i] and 
                volume_surge[i] and 
                trend_1d_up_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # Enter short: Donchian low breakout + volume surge + daily downtrend
            elif (close[i] < donchian_low[i] and 
                  volume_surge[i] and 
                  trend_1d_down_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit when price returns to Donchian mid or trend fails
            donchian_mid = (donchian_high[i] + donchian_low[i]) / 2
            if (close[i] < donchian_mid or 
                trend_1d_up_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit when price returns to Donchian mid or trend fails
            donchian_mid = (donchian_high[i] + donchian_low[i]) / 2
            if (close[i] > donchian_mid or 
                trend_1d_down_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals