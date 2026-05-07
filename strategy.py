#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R momentum with 1d trend filter and volume spike confirmation.
# Long when Williams %R crosses above -20 (oversold bounce) AND 1d EMA50 rising AND volume > 1.5x 20-period average.
# Short when Williams %R crosses below -80 (overbought rejection) AND 1d EMA50 falling AND volume > 1.5x 20-period average.
# Exit when Williams %R crosses back through -50 (middle level).
# This strategy captures momentum reversals at extremes with trend alignment and volume confirmation.
# Williams %R identifies overbought/oversold conditions. The 1d EMA50 filter ensures we trade with the daily trend.
# Volume spike confirms institutional participation. Target: 25-40 trades/year (100-160 total over 4 years).

name = "4h_WilliamsR_Momentum_1dEMA50_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Williams %R (14-period) - momentum oscillator
    # %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    period = 14
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    
    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 1d EMA50 direction
    ema50_rising = np.zeros_like(ema50_1d_aligned, dtype=bool)
    ema50_falling = np.zeros_like(ema50_1d_aligned, dtype=bool)
    ema50_rising[1:] = ema50_1d_aligned[1:] > ema50_1d_aligned[:-1]
    ema50_falling[1:] = ema50_1d_aligned[1:] < ema50_1d_aligned[:-1]
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20, 14)  # Sufficient warmup for EMA50, volume MA, and Williams %R
    
    for i in range(start_idx, n):
        if (np.isnan(williams_r[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(ema50_rising[i]) or np.isnan(ema50_falling[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Williams %R crosses above -20 (from below), 1d EMA50 rising, volume filter
            williams_r_cross_up = (williams_r[i] > -20) and (williams_r[i-1] <= -20)
            long_cond = williams_r_cross_up and ema50_rising[i] and volume_filter[i]
            
            # Short conditions: Williams %R crosses below -80 (from above), 1d EMA50 falling, volume filter
            williams_r_cross_down = (williams_r[i] < -80) and (williams_r[i-1] >= -80)
            short_cond = williams_r_cross_down and ema50_falling[i] and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Williams %R crosses below -50 (from above)
            if williams_r[i] < -50 and williams_r[i-1] >= -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R crosses above -50 (from below)
            if williams_r[i] > -50 and williams_r[i-1] <= -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals