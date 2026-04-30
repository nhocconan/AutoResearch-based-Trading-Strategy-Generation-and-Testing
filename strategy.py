#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 1d trend filter and volume confirmation.
# Williams %R identifies overbought/oversold conditions. In ranging markets, mean reversion from
# extreme readings works well. 1d EMA50 filters for higher timeframe trend direction to avoid
# counter-trend trades. Volume confirmation (>1.5x 20-bar average) ensures breakouts have
# conviction. Session filter (08-20 UTC) trades only during liquid hours. Discrete position
# sizing at ±0.25 manages fee drag. Target: 60-120 total trades over 4 years (15-30/year)
# to avoid excessive fees on 6h timeframe. Works in bull markets via trend-following mean
# reversion pullbacks and in bear markets via oversold bounces in downtrends.

name = "6h_WilliamsR_MeanRev_1dEMA50_Trend_VolumeConfirm_Session_v1"
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
    
    # Load 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d_vals = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d_vals).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 6h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams %R (14-period) on 6h: (Highest High - Close) / (Highest High - Lowest Low) * -100
    wr_period = 14
    highest_high = pd.Series(high).rolling(window=wr_period, min_periods=wr_period).max().values
    lowest_low = pd.Series(low).rolling(window=wr_period, min_periods=wr_period).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # warmup for Williams %R and EMA50
    
    for i in range(start_idx, n):
        # Skip if indicators not available or outside session
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(williams_r[i]) or
            np.isnan(volume_confirm[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_ema_50_1d = ema_50_1d_aligned[i]
        curr_williams_r = williams_r[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Williams %R oversold (< -80), price above 1d EMA50 (uptrend filter), volume confirmation
            if (curr_williams_r < -80 and 
                curr_close > curr_ema_50_1d and 
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20), price below 1d EMA50 (downtrend filter), volume confirmation
            elif (curr_williams_r > -20 and 
                  curr_close < curr_ema_50_1d and 
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position
            # Exit condition: Williams %R returns to neutral territory (> -50) or trend reverses
            if curr_williams_r > -50:  # Williams %R back above -50 = momentum fading
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit condition: Williams %R returns to neutral territory (< -50) or trend reverses
            if curr_williams_r < -50:  # Williams %R back below -50 = momentum fading
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals