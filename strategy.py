#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Williams %R (14) with 1-day trend filter (EMA34) and volume spike.
# Long when: Williams %R < -80 (oversold) AND EMA34(1d) rising AND volume > 1.8 * EMA20(volume).
# Short when: Williams %R > -20 (overbought) AND EMA34(1d) falling AND volume > 1.8 * EMA20(volume).
# Exit when Williams %R crosses back above -50 (for long) or below -50 (for short).
# Williams %R identifies mean-reversion extremes; EMA34 filters trend direction; volume spike confirms reversal.
# Designed for low trade frequency (target: 20-40/year) to minimize fee drag and improve generalization.
# Works in bull markets via buying oversold dips in uptrend and in bear markets via selling overbought rallies in downtrend.
name = "4h_WilliamsR_1dEMA34_VolumeSpike"
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
    
    # Williams %R (14) - momentum oscillator
    # %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    # Handle division by zero (when high == low)
    williams_r[highest_high == lowest_low] = -50  # neutral when no range
    
    # Williams %R thresholds
    williams_r_oversold = williams_r < -80
    williams_r_overbought = williams_r > -20
    williams_r_exit = williams_r > -50  # exit long when above -50
    williams_r_exit_short = williams_r < -50  # exit short when below -50
    
    # Load 1-day data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # EMA34 on 1d close
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    # Rising if current > previous, falling if current < previous
    ema_34_rising = np.zeros_like(ema_34_1d, dtype=bool)
    ema_34_falling = np.zeros_like(ema_34_1d, dtype=bool)
    ema_34_rising[1:] = ema_34_1d[1:] > ema_34_1d[:-1]
    ema_34_falling[1:] = ema_34_1d[1:] < ema_34_1d[:-1]
    
    ema_34_rising_aligned = align_htf_to_ltf(prices, df_1d, ema_34_rising)
    ema_34_falling_aligned = align_htf_to_ltf(prices, df_1d, ema_34_falling)
    
    # Volume confirmation: current volume > 1.8 * 20-period EMA of volume
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(williams_r[i]) or np.isnan(ema_34_rising_aligned[i]) or np.isnan(ema_34_falling_aligned[i]) or 
            np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Williams %R oversold AND EMA34(1d) rising AND volume spike
            long_condition = williams_r_oversold[i] and ema_34_rising_aligned[i] and volume_spike[i]
            # Short: Williams %R overbought AND EMA34(1d) falling AND volume spike
            short_condition = williams_r_overbought[i] and ema_34_falling_aligned[i] and volume_spike[i]
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Williams %R crosses above -50
            if williams_r_exit[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Williams %R crosses below -50
            if williams_r_exit_short[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals