#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R Extreme with 1d EMA34 trend filter and volume confirmation
# Williams %R identifies overbought/oversold conditions: > -20 = overbought, < -80 = oversold
# Extreme readings (< -90 for short, > -10 for long) indicate exhaustion points
# Only trade reversals from extremes in direction of 1d EMA34 trend (trade with higher timeframe trend)
# Volume spike (2.0x 20-period average) confirms institutional participation at turning points
# Works in bull markets via buying oversold bounces in uptrends and bear markets via selling overbought bounces in downtrends
# Discrete sizing 0.25 minimizes fee churn. Target: 75-200 total trades over 4 years (19-50/year).

name = "4h_WilliamsR_Extreme_1dEMA34_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC) to avoid datetime errors
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 1d data ONCE before loop (MTF Rule #1)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Williams %R (14-period)
    period_willr = 14
    highest_high = pd.Series(high).rolling(window=period_willr, min_periods=period_willr).max().values
    lowest_low = pd.Series(low).rolling(window=period_willr, min_periods=period_willr).min().values
    willr = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 34, 20)  # warmup for all indicators
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(willr[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
            
        # Session filter: only trade 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_ema_34_1d = ema_34_1d_aligned[i]
        curr_willr = willr[i]
        curr_volume_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike
            if curr_volume_spike:
                # Bullish entry: Williams %R oversold extreme (< -90) AND price above 1d EMA34 (uptrend)
                if curr_willr < -90 and curr_close > curr_ema_34_1d:
                    signals[i] = 0.25
                    position = 1
                # Bearish entry: Williams %R overbought extreme (> -10) AND price below 1d EMA34 (downtrend)
                elif curr_willr > -10 and curr_close < curr_ema_34_1d:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position
            # Exit when Williams %R returns to neutral territory (> -50) or price falls below EMA
            if curr_willr > -50 or curr_close < curr_ema_34_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when Williams %R returns to neutral territory (< -50) or price rises above EMA
            if curr_willr < -50 or curr_close > curr_ema_34_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals