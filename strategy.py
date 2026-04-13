#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Hypothesis: 6h Williams %R reversal + 12h trend filter + volume spike
    # Long when: Williams %R(14) crosses above -80 (oversold) AND 12h EMA50 > EMA200 (uptrend) AND volume > 2x 20-bar avg
    # Short when: Williams %R(14) crosses below -20 (overbought) AND 12h EMA50 < EMA200 (downtrend) AND volume > 2x 20-bar avg
    # Exit when: Williams %R crosses opposite extreme (-20 for long, -80 for short) OR trend filter fails
    # Uses discrete sizing (0.25) targeting 50-150 total trades over 4 years (12-37/year).
    # Williams %R identifies exhaustion points; 12h EMA filter ensures alignment with higher timeframe trend.
    # Volume spike confirms institutional participation at turning points.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 6h data for Williams %R (primary timeframe)
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 14:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Calculate Williams %R for 6h: (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_6h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_6h).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_6h) / (highest_high - lowest_low)
    # Handle division by zero (when highest_high == lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Align 6h Williams %R to 15m timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_6h, williams_r)
    
    # Get 12h data for EMA trend filter (HTF)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 200:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMAs for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200_12h = pd.Series(close_12h).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align 12h EMA values to 6h timeframe
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    ema_200_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_200_12h)
    
    # Trend filter: EMA50 > EMA200 for uptrend, EMA50 < EMA200 for downtrend
    uptrend = ema_50_12h_aligned > ema_200_12h_aligned
    downtrend = ema_50_12h_aligned < ema_200_12h_aligned
    
    # Volume confirmation: volume > 2x 20-bar average volume
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(200, n):  # Start after EMA200 warmup
        # Skip if data not ready
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_50_12h_aligned[i]) or np.isnan(ema_200_12h_aligned[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Williams %R crossover signals (using current bar vs previous bar)
        wr_cross_above_80 = (williams_r_aligned[i-1] <= -80) and (williams_r_aligned[i] > -80)
        wr_cross_below_20 = (williams_r_aligned[i-1] >= -20) and (williams_r_aligned[i] < -20)
        wr_cross_above_20 = (williams_r_aligned[i-1] <= -20) and (williams_r_aligned[i] > -20)
        wr_cross_below_80 = (williams_r_aligned[i-1] >= -80) and (williams_r_aligned[i] < -80)
        
        # Entry conditions
        long_entry = wr_cross_above_80 and uptrend[i] and volume_spike[i] and position != 1
        short_entry = wr_cross_below_20 and downtrend[i] and volume_spike[i] and position != -1
        
        # Exit conditions
        exit_long = position == 1 and (wr_cross_above_20 or not uptrend[i])
        exit_short = position == -1 and (wr_cross_below_80 or not downtrend[i])
        
        # Execute signals
        if long_entry:
            position = 1
            signals[i] = position_size
        elif short_entry:
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

name = "6h_12h_williamsr_ema_trend_volume_v1"
timeframe = "6h"
leverage = 1.0