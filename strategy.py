#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 1d EMA200 trend filter and volume confirmation.
# Williams %R identifies overbought/oversold conditions; EMA200 filters for higher timeframe trend direction.
# Volume confirmation (>1.8x 20-bar avg) reduces false signals. Session filter (08-20 UTC) avoids low liquidity.
# Discrete position sizing at ±0.25 to balance return and fee drag.
# Target: 80-120 total trades over 4 years (20-30/year) to stay within fee drag limits for 6h timeframe.
# Works in bull markets via buying dips in uptrend and in bear markets via selling rallies in downtrend.

name = "6h_WilliamsR_MeanRev_1dEMA200_Trend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours (08-20 UTC) to avoid look-ahead
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 1d data ONCE before loop for EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 210:
        return np.zeros(n)
    
    # Calculate 1d EMA200 for trend filter
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Williams %R (14-period) on 6h data
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close) / (highest_high_14 - lowest_low_14)
    # Handle division by zero when high == low
    williams_r = np.where((highest_high_14 - lowest_low_14) == 0, -50, williams_r)
    
    # Volume confirmation: volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.8 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # warmup for EMA200 and Williams %R
    
    for i in range(start_idx, n):
        # Skip if indicators not available or outside session
        if (np.isnan(ema_200_1d_aligned[i]) or 
            np.isnan(williams_r[i]) or np.isnan(volume_confirm[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_ema_200_1d = ema_200_1d_aligned[i]
        curr_williams_r = williams_r[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Williams %R oversold (< -80), price > 1d EMA200 (uptrend), volume confirmation
            if (curr_williams_r < -80 and 
                curr_close > curr_ema_200_1d and 
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20), price < 1d EMA200 (downtrend), volume confirmation
            elif (curr_williams_r > -20 and 
                  curr_close < curr_ema_200_1d and 
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position
            # Exit: Williams %R returns to neutral (> -50) or price crosses below EMA200
            if (curr_williams_r > -50 or 
                curr_close < curr_ema_200_1d):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Williams %R returns to neutral (< -50) or price crosses above EMA200
            if (curr_williams_r < -50 or 
                curr_close > curr_ema_200_1d):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals