#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Williams %R with 4h EMA50 trend filter and 1d volume confirmation
# Williams %R(14) identifies overbought/oversold conditions for mean reversion entries
# 4h EMA50 ensures alignment with higher timeframe trend direction
# 1d volume spike (>1.8x 20-period EMA volume) confirms institutional participation
# Discrete sizing 0.20 targets 60-150 total trades over 4 years (15-37/year) for 1h timeframe
# Works in bull markets (buy oversold in uptrend) and bear markets (sell overbought in downtrend)
# Session filter (08-20 UTC) reduces noise trades during low-liquidity periods

name = "1h_WilliamsR_4hEMA50_1dVolumeSpike_Session"
timeframe = "1h"
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
    
    # Get 4h data for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_shifted = np.roll(ema50_4h, 1)
    ema50_4h_shifted[0] = np.nan
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h_shifted)
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    vol_ema_20_1d = pd.Series(volume_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ema_20_1d)
    
    # Williams %R(14) on 1h: (Highest High - Close) / (Highest High - Lowest Low) * -100
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    highest_high = high_series.rolling(window=14, min_periods=14).max().values
    lowest_low = low_series.rolling(window=14, min_periods=14).min().values
    williams_r = ((highest_high - close) / (highest_high - lowest_low)) * -100
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema50_4h_aligned[i]) or np.isnan(vol_ema_20_1d_aligned[i]) or 
            np.isnan(williams_r[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Williams %R < -80 (oversold) AND 4h EMA50 uptrend AND 1d volume spike
            if williams_r[i] < -80 and close[i] > ema50_4h_aligned[i] and volume[i] > (1.8 * vol_ema_20_1d_aligned[i]):
                signals[i] = 0.20
                position = 1
            # Short conditions: Williams %R > -20 (overbought) AND 4h EMA50 downtrend AND 1d volume spike
            elif williams_r[i] > -20 and close[i] < ema50_4h_aligned[i] and volume[i] > (1.8 * vol_ema_20_1d_aligned[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: Williams %R > -50 (exiting oversold) OR 4h EMA50 turns down
            if williams_r[i] > -50 or close[i] < ema50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: Williams %R < -50 (exiting overbought) OR 4h EMA50 turns up
            if williams_r[i] < -50 or close[i] > ema50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals