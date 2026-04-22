#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R reversal with 12h trend filter and volume confirmation.
Long when Williams %R crosses above -80 (oversold) with bullish 12h trend and volume spike.
Short when Williams %R crosses below -20 (overbought) with bearish 12h trend and volume spike.
Exit when Williams %R returns to -50 (mean reversion zone).
Williams %R identifies exhaustion points in trends; 12h filter avoids counter-trend trades.
Designed for low trade frequency (10-25/year) to minimize fee drag in ranging markets.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data for trend filter - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Williams %R (14-period) on 6h
    lookback = 14
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max()
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min()
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    williams_r = williams_r.replace([np.inf, -np.inf], np.nan).values
    
    # 12h EMA50 trend filter
    close_12h = pd.Series(df_12h['close'].values)
    ema_50_12h = close_12h.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # 12h close for trend direction (bullish if close > EMA50)
    close_12h_aligned = align_htf_to_ltf(prices, df_12h, df_12h['close'].values)
    bullish_trend = close_12h_aligned > ema_50_12h_aligned
    
    # 6h volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(lookback, n):
        # Skip if data not ready
        if (np.isnan(williams_r[i]) or np.isnan(bullish_trend[i]) or 
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Williams %R crosses above -80 (oversold) with bullish 12h trend and volume
            if (williams_r[i] > -80 and williams_r[i-1] <= -80 and  # Cross above -80
                bullish_trend[i] and 
                volume[i] > 1.8 * vol_avg_20[i]):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 (overbought) with bearish 12h trend and volume
            elif (williams_r[i] < -20 and williams_r[i-1] >= -20 and  # Cross below -20
                  not bullish_trend[i] and 
                  volume[i] > 1.8 * vol_avg_20[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit when Williams %R returns to -50 (mean reversion)
            if position == 1 and williams_r[i] >= -50:
                signals[i] = 0.0
                position = 0
            elif position == -1 and williams_r[i] <= -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_WilliamsR_Reversal_12hTrend_Volume"
timeframe = "6h"
leverage = 1.0
#%%