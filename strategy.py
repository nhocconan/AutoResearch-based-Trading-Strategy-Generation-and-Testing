#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R breakout with 12h EMA50 trend and volume spike filter.
# Long when: Williams %R crosses above -20 from below AND 12h close > 12h EMA50 AND 4h volume > 2.0x 20-period average
# Short when: Williams %R crosses below -80 from above AND 12h close < 12h EMA50 AND 4h volume > 2.0x 20-period average
# Uses Williams %R for momentum extremes, 12h EMA50 for trend alignment, volume spike for conviction.
# Target: 20-40 trades/year on 4h. Discrete sizing 0.25 to minimize fee drag while capturing momentum.
# Works in bull (breakouts with trend) and bear (breakdowns with trend) by trading with aligned 12h trend.

name = "4h_WilliamsR_12hEMA50_VolumeSpike_v1"
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
    
    # Pre-compute session hours for efficiency
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # Load 12h data ONCE before loop for Williams %R calculation and trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 14:
        return np.zeros(n)
    
    # Calculate 12h Williams %R (14-period)
    highest_high = pd.Series(df_12h['high'].values).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(df_12h['low'].values).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - df_12h['close'].values) / (highest_high - lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)  # handle division by zero
    
    # Align Williams %R to 4h primary timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_12h, williams_r)
    
    # 12h EMA50 for trend filter
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # 4h volume average (20-period) for volume confirmation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    vol_4h = df_4h['volume'].values
    vol_ma_4h = pd.Series(vol_4h).rolling(window=20, min_periods=20).mean().values
    vol_ma_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for 12h EMA50 (need 50+14 for Williams %R)
    
    for i in range(start_idx, n):
        # Session filter: 08-20 UTC (reduce noise, focus on active sessions)
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            # Outside session: flatten position if any
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Skip if any data not ready
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_50_aligned[i]) or 
            np.isnan(vol_ma_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_vol = volume[i]
        curr_vol_ma = vol_ma_4h_aligned[i]
        curr_williams_r = williams_r_aligned[i]
        curr_ema_50 = ema_50_aligned[i]
        
        # Williams %R crossover signals
        williams_cross_above = (curr_williams_r > -20) and (i > start_idx and williams_r_aligned[i-1] <= -20)
        williams_cross_below = (curr_williams_r < -80) and (i > start_idx and williams_r_aligned[i-1] >= -80)
        
        # Volume confirmation: current 4h volume > 2.0x 20-period average
        volume_confirm = curr_vol > (curr_vol_ma * 2.0)
        
        # 12h trend filter
        uptrend_12h = curr_close > curr_ema_50
        downtrend_12h = curr_close < curr_ema_50
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: Williams %R crosses above -20 AND 12h uptrend AND volume confirmation
            if williams_cross_above and uptrend_12h and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -80 AND 12h downtrend AND volume confirmation
            elif williams_cross_below and downtrend_12h and volume_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Williams %R crosses below -80 OR 12h trend turns down
            if williams_cross_below or not uptrend_12h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Williams %R crosses above -20 OR 12h trend turns up
            if williams_cross_above or not downtrend_12h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals