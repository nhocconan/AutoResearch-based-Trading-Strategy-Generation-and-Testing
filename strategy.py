#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Williams %R extreme + weekly EMA trend filter + volume confirmation
    # Uses 1w for primary trend direction (major market regime), 6h for entry timing
    # Williams %R(14) < -80 for oversold long, > -20 for overbought short
    # Weekly EMA50 trend filter ensures trading with major trend to avoid whipsaws
    # Volume spike (>2.0x 24-period average) confirms institutional participation
    # Target: 12-25 trades/year (50-100 total over 4 years) to minimize fee drag
    # Works in both bull/bear: buys extreme fear in uptrends, sells extreme greed in downtrends
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA50 for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Williams %R on 6h data
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    lookback = 14
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    # Handle division by zero (when highest_high == lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Get 6h volume for confirmation (>2.0x 24-period average)
    vol_ma_6h = np.full(n, np.nan)
    for i in range(24, n):
        vol_ma_6h[i] = np.mean(volume[i-24:i])
    volume_spike_6h = volume > (2.0 * vol_ma_6h)
    
    # Align all indicators to LTF (6h)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(volume_spike_6h[i])):
            signals[i] = 0.0
            continue
        
        # Williams %R extreme conditions
        oversold = williams_r[i] < -80
        overbought = williams_r[i] > -20
        
        # Weekly trend filter
        bullish_trend = close[i] > ema50_1w_aligned[i]
        bearish_trend = close[i] < ema50_1w_aligned[i]
        
        # Entry logic: Extreme + trend alignment + volume confirmation
        long_entry = oversold and bullish_trend and volume_spike_6h[i]
        short_entry = overbought and bearish_trend and volume_spike_6h[i]
        
        # Exit logic: Williams %R returns to neutral zone (-50) or trend reversal
        long_exit = williams_r[i] > -50 or not bullish_trend
        short_exit = williams_r[i] < -50 or not bearish_trend
        
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

name = "6h_1w_williamsr_extreme_ema50_volume_v1"
timeframe = "6h"
leverage = 1.0