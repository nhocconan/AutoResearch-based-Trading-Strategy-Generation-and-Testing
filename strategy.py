#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h timeframe with 1d Williams %R momentum and 1w EMA trend filter.
# Long: Williams %R crosses above -50 (momentum shift) + price > 1w EMA50 + volume > 1.3x avg volume (20-period).
# Short: Williams %R crosses below -50 + price < 1w EMA50 + volume > 1.3x avg volume.
# Uses 1d for momentum signal, 1w for trend filter, 12h for execution with volume confirmation.
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for Williams %R (14-period)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = np.full(len(high_1d), np.nan)
    lowest_low = np.full(len(low_1d), np.nan)
    williams_r = np.full(len(close_1d), np.nan)
    
    for i in range(14, len(high_1d)):
        highest_high[i] = np.max(high_1d[i-14:i])
        lowest_low[i] = np.min(low_1d[i-14:i])
        hh_ll = highest_high[i] - lowest_low[i]
        if hh_ll != 0:
            williams_r[i] = ((highest_high[i] - close_1d[i]) / hh_ll) * -100
        else:
            williams_r[i] = -50  # neutral when range is zero
    
    # 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Average volume (20-period) for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    # Align 1d Williams %R to 12h
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Align 1w EMA50 to 12h
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        # Skip if any required data is not ready
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        wr = williams_r_aligned[i]
        ema_trend = ema_50_1w_aligned[i]
        
        # Williams %R crossover signals
        wr_cross_above = (wr > -50) and (i == 20 or williams_r_aligned[i-1] <= -50)
        wr_cross_below = (wr < -50) and (i == 20 or williams_r_aligned[i-1] >= -50)
        
        # Volume confirmation: current volume > 1.3x average volume
        volume_confirm = vol > 1.3 * avg_vol
        
        if position == 0:
            # Long: Williams %R crosses above -50 + price above EMA50 + volume confirmation
            if wr_cross_above and (price > ema_trend) and volume_confirm:
                position = 1
                signals[i] = position_size
            # Short: Williams %R crosses below -50 + price below EMA50 + volume confirmation
            elif wr_cross_below and (price < ema_trend) and volume_confirm:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Williams %R crosses below -50 or price below EMA50
            if wr_cross_below or (price < ema_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Williams %R crosses above -50 or price above EMA50
            if wr_cross_above or (price > ema_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1d_1w_WilliamsR_EMA_Volume"
timeframe = "12h"
leverage = 1.0