#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams %R mean reversion with 1w EMA200 trend filter and volume spike confirmation
# Williams %R identifies overbought/oversold conditions for mean reversion entries
# 1w EMA200 provides primary trend filter to avoid counter-trend trades in strong trends
# Volume spike (>2.0x 20-period EMA) confirms institutional participation
# Designed for 1d timeframe targeting 30-100 trades over 4 years (7-25/year)
# Works in bull markets (oversold bounces in uptrend) and bear markets (overbought selloffs in downtrend)
# Williams %R is effective in ranging/volatile markets which occur in both regimes

name = "1d_WilliamsR_MeanReversion_1wEMA200_Trend_Volume"
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
    
    # Get 1w data for EMA200 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    # Calculate 1w EMA200
    close_1w = df_1w['close'].values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Calculate Williams %R (14-period)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    # Handle division by zero (when highest_high == lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Get 1d data for volume EMA
    df_1d = get_htf_data(prices, '1d')
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
        if (np.isnan(ema_200_1w_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(vol_ema_20_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current 1d volume > 2.0 x 20-period EMA
        volume_confirmed = volume[i] > (2.0 * vol_ema_20_aligned[i])
        
        if position == 0:
            # Long: Williams %R oversold (< -80) + volume confirmation + 1w EMA > EMA200 (uptrend)
            if (williams_r[i] < -80 and volume_confirmed and 
                close[i] > ema_200_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20) + volume confirmation + 1w EMA < EMA200 (downtrend)
            elif (williams_r[i] > -20 and volume_confirmed and 
                  close[i] < ema_200_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R rises above -50 (mean reversion) OR 1w EMA < EMA200 (trend change)
            if williams_r[i] > -50 or close[i] < ema_200_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R falls below -50 (mean reversion) OR 1w EMA > EMA200 (trend change)
            if williams_r[i] < -50 or close[i] > ema_200_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals