#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R extreme reversal with 1d EMA50 trend filter and volume confirmation
# Williams %R < -80 = oversold, > -20 = overbought. Reversal signals when price rejects extremes
# 1d EMA50 filter ensures we trade counter-trend to 6h extreme but in direction of daily trend
# Volume spike (1.8x 24-period average) confirms participation and reduces false signals
# Works in bull markets via buying dips in uptrends and bear markets via selling rallies in downtrends
# Discrete sizing 0.25 minimizes fee churn. Target: 60-120 total trades over 4 years (15-30/year).

name = "6h_WilliamsR_Extreme_1dEMA50_VolumeConfirm_v1"
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
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC) to avoid datetime errors
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 1d data ONCE before loop (MTF Rule #1)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams %R (14-period) on 6h data
    lookback = 14
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # Volume confirmation: volume > 1.8x 24-period average
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (1.8 * vol_ma_24)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, lookback, 50, 24)  # warmup
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(vol_ma_24[i])):
            signals[i] = 0.0
            continue
            
        # Session filter: only trade 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_williams_r = williams_r[i]
        curr_ema_50_1d = ema_50_1d_aligned[i]
        curr_volume_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike
            if curr_volume_spike:
                # Bullish entry: Williams %R < -80 (oversold) AND price > 1d EMA50 (uptrend)
                if curr_williams_r < -80 and curr_close > curr_ema_50_1d:
                    signals[i] = 0.25
                    position = 1
                # Bearish entry: Williams %R > -20 (overbought) AND price < 1d EMA50 (downtrend)
                elif curr_williams_r > -20 and curr_close < curr_ema_50_1d:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position
            # Exit when Williams %R > -20 (overbought) or price < 1d EMA50 (trend break)
            if curr_williams_r > -20 or curr_close < curr_ema_50_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when Williams %R < -80 (oversold) or price > 1d EMA50 (trend break)
            if curr_williams_r < -80 or curr_close > curr_ema_50_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals