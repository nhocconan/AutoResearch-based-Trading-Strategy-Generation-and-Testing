#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams %R extreme with 1w EMA34 trend filter and volume confirmation
# Williams %R identifies overbought/oversold conditions; extreme readings (< -80 or > -20) 
# combined with weekly trend alignment provide high-probability mean reversion entries.
# Volume spike confirms institutional participation at these extremes.
# Designed for low trade frequency (target: 7-25/year) on 1d timeframe to minimize fee drag.
# Works in both bull and bear markets by trading with/against the weekly trend appropriately.

name = "1d_WilliamsR_Extreme_1wEMA34_VolumeConfirm"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1w data for Williams %R calculation, EMA, and volume
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 40:
        return np.zeros(n)
    
    # Calculate 1w Williams %R (14-period)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(df_1w['high'].values).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(df_1w['low'].values).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - df_1w['close'].values) / (highest_high - lowest_low + 1e-10) * -100
    
    # Calculate 1w EMA34 for trend filter
    ema_34 = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 1w volume spike (volume > 1.8 * 20-period EMA of volume)
    vol_ema_20 = pd.Series(df_1w['volume'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = df_1w['volume'].values > (1.8 * vol_ema_20)
    
    # Align 1w indicators to 1d timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1w, williams_r)
    ema_34_aligned = align_htf_to_ltf(prices, df_1w, ema_34)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1w, volume_spike)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(40, n):
        # Skip if any value is NaN or outside session
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_34_aligned[i]) or 
            np.isnan(volume_spike_aligned[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine weekly trend direction
        is_uptrend = close[i] > ema_34_aligned[i]
        is_downtrend = close[i] < ema_34_aligned[i]
        
        if position == 0:
            # Long: Williams %R extremely oversold (< -80) in any trend with volume spike
            # More aggressive in uptrend (buy the dip), cautious in downtrend
            if williams_r_aligned[i] < -80 and volume_spike_aligned[i]:
                if is_uptrend:
                    signals[i] = 0.30  # Full position in uptrend
                    position = 1
                else:
                    signals[i] = 0.15  # Half position in downtrend (mean reversion)
                    position = 1
            # Short: Williams %R extremely overbought (> -20) in any trend with volume spike
            # More aggressive in downtrend (sell the rally), cautious in uptrend
            elif williams_r_aligned[i] > -20 and volume_spike_aligned[i]:
                if is_downtrend:
                    signals[i] = -0.30  # Full position in downtrend
                    position = -1
                else:
                    signals[i] = -0.15  # Half position in uptrend (mean reversion)
                    position = -1
        elif position == 1:
            # Exit long: Williams %R returns to neutral territory (> -50) or reaches extreme overbought
            if williams_r_aligned[i] > -50 or williams_r_aligned[i] > -20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30 if is_uptrend else 0.15
        elif position == -1:
            # Exit short: Williams %R returns to neutral territory (< -50) or reaches extreme oversold
            if williams_r_aligned[i] < -50 or williams_r_aligned[i] < -80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30 if is_downtrend else -0.15
    
    return signals