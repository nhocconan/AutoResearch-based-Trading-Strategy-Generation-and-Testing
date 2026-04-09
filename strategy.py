#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Williams %R extremes + 1w trend filter + volume confirmation
# Williams %R(14) on 1d identifies overbought/oversold conditions
# Long when %R < -80 (oversold) and price > 1w EMA50 (bullish trend)
# Short when %R > -20 (overbought) and price < 1w EMA50 (bearish trend)
# Volume confirmation: current 6h volume > 1.5x average 6h volume (20-period)
# Discrete position sizing 0.25 to target ~20-40 trades/year and minimize fee drag
# Works in bull/bear markets: mean reversion in extremes + trend filter avoids counter-trend trades

name = "6h_1d_1w_williamsr_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Williams %R (14-period)
    def rolling_max(arr, window):
        return pd.Series(arr).rolling(window=window, min_periods=window).max().values
    
    def rolling_min(arr, window):
        return pd.Series(arr).rolling(window=window, min_periods=window).min().values
    
    highest_high_14 = rolling_max(high_1d, 14)
    lowest_low_14 = rolling_min(low_1d, 14)
    williams_r = -100 * (highest_high_14 - close_1d) / (highest_high_14 - lowest_low_14)
    williams_r = np.where((highest_high_14 - lowest_low_14) == 0, -50, williams_r)  # avoid division by zero
    
    # Load 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Calculate 1w EMA (50-period)
    close_1w_series = pd.Series(close_1w)
    ema_50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d Williams %R to 6h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Align 1w EMA50 to 6h timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Pre-compute volume confirmation (20-period average on 6h)
    vol_s = pd.Series(volume)
    avg_vol_20 = vol_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(avg_vol_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 1.5x average 6h volume (20-period)
        volume_confirmed = volume[i] > 1.5 * avg_vol_20[i]
        
        if position == 1:  # Long position
            # Exit long if Williams %R rises above -50 (exiting oversold) or trend turns bearish
            if williams_r_aligned[i] > -50 or close[i] < ema_50_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit short if Williams %R falls below -50 (exiting overbought) or trend turns bullish
            if williams_r_aligned[i] < -50 or close[i] > ema_50_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Mean reversion strategy: enter at Williams %R extremes with trend filter and volume confirmation
            if williams_r_aligned[i] < -80 and close[i] > ema_50_1w_aligned[i] and volume_confirmed:
                position = 1
                signals[i] = 0.25
            elif williams_r_aligned[i] > -20 and close[i] < ema_50_1w_aligned[i] and volume_confirmed:
                position = -1
                signals[i] = -0.25
    
    return signals