#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R mean reversion with 12h trend filter and volume confirmation.
Long when %R crosses above -80 (oversold) with 12h bullish trend and volume spike.
Short when %R crosses below -20 (overbought) with 12h bearish trend and volume spike.
Exit when %R crosses -50 (mean reversion center) or trend weakens.
Williams %R identifies overextended moves in ranging markets, while 12h trend filter avoids
counter-trend trades during strong moves. Designed for low trade frequency (15-35/year).
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data for trend filter - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate Williams %R (14-period) on 6h
    lookback = 14
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # Calculate 12h EMA (34-period) for trend filter
    ema_12h = pd.Series(df_12h['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Calculate 12h price change rate for trend strength
    price_change_12h = pd.Series(df_12h['close'].values).pct_change(periods=2).values  # 2-period change (~1 day)
    price_change_aligned = align_htf_to_ltf(prices, df_12h, price_change_12h)
    
    # Calculate 6h volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(lookback, n):
        # Skip if data not ready
        if (np.isnan(williams_r[i]) or np.isnan(ema_12h_aligned[i]) or 
            np.isnan(price_change_aligned[i]) or np.isnan(vol_avg_20[i])):
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
            # Long: Williams %R crosses above -80 (oversold recovery) with bullish 12h trend and volume
            if (williams_r[i] > -80 and williams_r[i-1] <= -80 and  # Cross above -80
                ema_12h_aligned[i] > ema_12h_aligned[i-1] and      # 12h EMA rising
                price_change_aligned[i] > 0 and                    # Positive 12h momentum
                volume[i] > 1.8 * vol_avg_20[i]):                  # Volume spike
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 (overbought decline) with bearish 12h trend and volume
            elif (williams_r[i] < -20 and williams_r[i-1] >= -20 and  # Cross below -20
                  ema_12h_aligned[i] < ema_12h_aligned[i-1] and      # 12h EMA falling
                  price_change_aligned[i] < 0 and                    # Negative 12h momentum
                  volume[i] > 1.8 * vol_avg_20[i]):                  # Volume spike
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: %R crosses below -50 OR 12h trend turns bearish
                if williams_r[i] < -50 or ema_12h_aligned[i] < ema_12h_aligned[i-1]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: %R crosses above -50 OR 12h trend turns bullish
                if williams_r[i] > -50 or ema_12h_aligned[i] > ema_12h_aligned[i-1]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_WilliamsR_12hTrendFilter_Volume"
timeframe = "6h"
leverage = 1.0
#%%