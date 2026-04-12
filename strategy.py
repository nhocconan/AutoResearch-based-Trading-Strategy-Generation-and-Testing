#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h Williams %R extreme + 1d EMA200 trend filter + volume confirmation
    # Williams %R(14) < -80 = oversold (long), > -20 = overbought (short) on 12h
    # 1d EMA200 filter: only long when price > EMA200, short when price < EMA200
    # Volume confirmation: 12h volume > 1.5x 20-period average
    # Session filter: 08-20 UTC to avoid low-liquidity hours
    # Discrete position sizing: ±0.25 to minimize fee churn
    # Target: 12-25 trades/year (50-100 total over 4 years) to stay within fee drag limits
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA200 for trend filter
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Calculate 12h Williams %R(14)
    period = 14
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero (when highest_high == lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Get 12h volume for confirmation (>1.5x 20-period average)
    vol_ma_12h = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_12h[i] = np.mean(volume[i-20:i])
    volume_spike_12h = volume > (1.5 * vol_ma_12h)
    
    # Align all indicators to LTF (12h)
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready or outside session
        if (np.isnan(ema200_1d_aligned[i]) or np.isnan(williams_r_aligned[i]) or 
            np.isnan(volume_spike_12h[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Williams %R extremes
        oversold = williams_r_aligned[i] < -80
        overbought = williams_r_aligned[i] > -20
        
        # 1d EMA200 trend filter
        bullish_trend = close[i] > ema200_1d_aligned[i]
        bearish_trend = close[i] < ema200_1d_aligned[i]
        
        # Entry logic: Extreme + trend alignment + volume confirmation
        long_entry = oversold and bullish_trend and volume_spike_12h[i]
        short_entry = overbought and bearish_trend and volume_spike_12h[i]
        
        # Exit logic: Williams %R returns to neutral zone (-50) or trend fails
        williams_r_neutral = abs(williams_r_aligned[i] + 50) < 20  # within ±20 of -50
        trend_failed = (position == 1 and not bullish_trend) or (position == -1 and not bearish_trend)
        
        long_exit = williams_r_neutral or trend_failed
        short_exit = williams_r_neutral or trend_failed
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_1d_williamsr_extreme_ema200_volume_v1"
timeframe = "12h"
leverage = 1.0