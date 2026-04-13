#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 1d Williams %R mean reversion with 1w trend filter and volume confirmation
    # Works in both bull and bear: Williams %R captures overbought/oversold reversals,
    # 1w trend filters counter-trend noise, volume confirms momentum
    # Target: 20-40 trades/year to minimize fee drag
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values if 'volume' in prices.columns else np.ones(len(prices))
    
    # Get 1d data for Williams %R calculation (primary timeframe)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values if 'volume' in df_1d.columns else np.ones(len(df_1d))
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate Williams %R (14-period) on 1d
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close_1d) / (highest_high_14 - lowest_low_14)
    
    # Calculate 1w EMA(20) for trend filter
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate 1d volume confirmation (20-period average)
    vol_avg_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all HTF indicators to 1d primary timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(williams_r_aligned[i]) or 
            np.isnan(ema_20_1w_aligned[i]) or
            np.isnan(vol_avg_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.2x 20-period average
        idx_1d = i // 24  # 24 hours in a day for 1h bars, but we use 1d data so adjust
        if idx_1d >= len(volume_1d):
            signals[i] = 0.0
            continue
        volume_confirmed = volume_1d[idx_1d] > 1.2 * vol_avg_20_1d_aligned[i]
        
        # Williams %R conditions: oversold < -80, overbought > -20
        oversold = williams_r_aligned[i] < -80
        overbought = williams_r_aligned[i] > -20
        
        # Trend direction from 1w EMA(20)
        # Need to get current 1w close price for trend comparison
        idx_1w = i // (24 * 7)  # approximate weeks
        if idx_1w >= len(close_1w):
            signals[i] = 0.0
            continue
        trend_up = close_1w[idx_1w] > ema_20_1w_aligned[i]
        trend_down = close_1w[idx_1w] < ema_20_1w_aligned[i]
        
        # Entry conditions: Williams %R extreme + trend alignment + volume
        enter_long = oversold and trend_up and volume_confirmed
        enter_short = overbought and trend_down and volume_confirmed
        
        # Exit conditions: Williams %R returns to neutral zone (-50)
        exit_long = position == 1 and williams_r_aligned[i] > -50
        exit_short = position == -1 and williams_r_aligned[i] < -50
        
        # Execute signals
        if enter_long and position != 1:
            position = 1
            signals[i] = position_size
        elif enter_short and position != -1:
            position = -1
            signals[i] = -position_size
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_1w_williamsr_meanreversion_trend_volume_v1"
timeframe = "1d"
leverage = 1.0