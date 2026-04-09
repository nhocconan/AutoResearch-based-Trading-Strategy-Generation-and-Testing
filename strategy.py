#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Williams %R extremes with 1w trend filter and volume confirmation
# Williams %R(14) on 1d identifies overbought/oversold conditions
# Long when %R crosses above -80 from below AND price > 1w EMA200 (uptrend filter)
# Short when %R crosses below -20 from above AND price < 1w EMA200 (downtrend filter)
# Volume confirmation: current 6h volume > 1.8x average 6h volume (20-period)
# Discrete position sizing 0.25 targets ~20-40 trades/year to minimize fee drag
# Works in bull/bear markets: mean reversion in ranges, trend filter avoids counter-trend trades

name = "6h_1d_1w_williamsr_trend_v2"
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
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d Williams %R (14-period)
    def williams_r(high_arr, low_arr, close_arr, window):
        highest_high = pd.Series(high_arr).rolling(window=window, min_periods=window).max().values
        lowest_low = pd.Series(low_arr).rolling(window=window, min_periods=window).min().values
        wr = -100 * (highest_high - close_arr) / (highest_high - lowest_low)
        return wr
    
    wr_14 = williams_r(high_1d, low_1d, close_1d, 14)
    
    # Calculate 1w EMA200 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align 1d indicators to 6h timeframe
    wr_14_aligned = align_htf_to_ltf(prices, df_1d, wr_14)
    
    # Align 1w EMA200 to 6h timeframe
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Pre-compute session filter (08-20 UTC) - optional but helps reduce noise
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(wr_14_aligned[i]) or np.isnan(ema_200_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 1.8x average 6h volume (20-period)
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_confirmed = volume[i] > 1.8 * vol_ma_20[i] if not np.isnan(vol_ma_20[i]) else False
        
        if position == 1:  # Long position
            # Exit long if Williams %R crosses below -50 (momentum loss) or price < 1w EMA200
            if wr_14_aligned[i] < -50 or close[i] < ema_200_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit short if Williams %R crosses above -50 (momentum loss) or price > 1w EMA200
            if wr_14_aligned[i] > -50 or close[i] > ema_200_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Mean reversion entries with trend filter and volume confirmation
            # Long: Williams %R crosses above -80 from below (oversold bounce) in uptrend
            wr_long_signal = (wr_14_aligned[i] > -80) and (wr_14_aligned[i-1] <= -80) if i > 0 else False
            wr_short_signal = (wr_14_aligned[i] < -20) and (wr_14_aligned[i-1] >= -20) if i > 0 else False
            
            if wr_long_signal and close[i] > ema_200_1w_aligned[i] and volume_confirmed:
                position = 1
                signals[i] = 0.25
            elif wr_short_signal and close[i] < ema_200_1w_aligned[i] and volume_confirmed:
                position = -1
                signals[i] = -0.25
    
    return signals