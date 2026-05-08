#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R with 1d trend filter and volume confirmation.
# Long when Williams %R(14) crosses above -80 (oversold) AND 1d EMA50 uptrend AND 1d volume > 1.5x 20-day average.
# Short when Williams %R(14) crosses below -20 (overbought) AND 1d EMA50 downtrend AND 1d volume > 1.5x 20-day average.
# Exit when Williams %R crosses back to neutral zone (-50).
# Uses 4h timeframe with 1d trend and volume filters to reduce false signals and control trade frequency.
# Target: 100-200 total trades over 4 years (25-50/year) with controlled frequency to avoid fee drag.

name = "4h_WilliamsR_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Daily data for trend and volume filters
    df_d = get_htf_data(prices, '1d')
    if len(df_d) < 50:
        return np.zeros(n)
    
    # Williams %R(14) on 4h data
    williams_period = 14
    highest_high = pd.Series(high).rolling(window=williams_period, min_periods=williams_period).max().values
    lowest_low = pd.Series(low).rolling(window=williams_period, min_periods=williams_period).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    williams_r[np.isnan(williams_r)] = -50  # Neutral when undefined
    
    # Daily EMA50 for trend direction
    close_d = df_d['close'].values
    ema50_d = pd.Series(close_d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_prev = np.roll(ema50_d, 1)
    ema50_prev[0] = ema50_d[0]
    ema50_uptrend = ema50_d > ema50_prev
    ema50_downtrend = ema50_d < ema50_prev
    
    # Daily volume filter: current volume > 1.5x 20-day average
    volume_d = df_d['volume'].values
    vol_ma20_d = pd.Series(volume_d).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_filter_d = volume_d > (1.5 * vol_ma20_d)
    
    # Align daily indicators to 4h timeframe
    ema50_uptrend_aligned = align_htf_to_ltf(prices, df_d, ema50_uptrend)
    ema50_downtrend_aligned = align_htf_to_ltf(prices, df_d, ema50_downtrend)
    volume_filter_aligned = align_htf_to_ltf(prices, df_d, volume_filter_d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(williams_period, 50)  # Sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(williams_r[i]) or 
            np.isnan(ema50_uptrend_aligned[i]) or np.isnan(ema50_downtrend_aligned[i]) or
            np.isnan(volume_filter_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Williams %R crosses above -80 (from below), uptrend, high volume
            williams_cross_up = (williams_r[i-1] <= -80) and (williams_r[i] > -80)
            long_cond = williams_cross_up and ema50_uptrend_aligned[i] and volume_filter_aligned[i]
            
            # Short conditions: Williams %R crosses below -20 (from above), downtrend, high volume
            williams_cross_down = (williams_r[i-1] >= -20) and (williams_r[i] < -20)
            short_cond = williams_cross_down and ema50_downtrend_aligned[i] and volume_filter_aligned[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Williams %R crosses back above -50 (exiting oversold)
            if williams_r[i-1] <= -50 and williams_r[i] > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R crosses back below -50 (exiting overbought)
            if williams_r[i-1] >= -50 and williams_r[i] < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals