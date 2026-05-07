#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Camarilla pivot breakout with 1-day trend filter (EMA34) and volume confirmation.
# Long when: Close > Camarilla R3 level AND EMA34(1d) rising AND volume > 2.0 * EMA20(volume).
# Short when: Close < Camarilla S3 level AND EMA34(1d) falling AND volume > 2.0 * EMA20(volume).
# Exit when price crosses back below/above Camarilla R1/S1 levels.
# Designed for low trade frequency (target: 20-30/year) to minimize fee drag and improve generalization.
# Camarilla levels provide precise support/resistance; daily trend filter ensures alignment with higher timeframe momentum.
# Works in bull markets via upward breakouts and in bear markets via downward breakouts.
name = "4h_Camarilla_R3S3_1dEMA34_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Camarilla Pivot Levels: calculated from previous day's OHLC
    # R3 = close + (high - low) * 1.1/2
    # S3 = close - (high - low) * 1.1/2
    # R1 = close + (high - low) * 1.1/12
    # S1 = close - (high - low) * 1.1/12
    # We need previous day's data, so we shift by 1 day (96 periods for 4h data)
    shift_periods = 96  # 24 hours * 60 minutes / 15 minutes per period (but we're on 4h, so 24/4 = 6 days? No)
    # Actually: 1 day = 24 hours, 4h bars per day = 6, so to get previous day's OHLC we need to look back 6 periods
    shift_periods = 6  # 6 * 4h = 24 hours = 1 day
    
    # Shift high, low, close by 6 periods to get previous day's values
    prev_high = np.roll(high, shift_periods)
    prev_low = np.roll(low, shift_periods)
    prev_close = np.roll(close, shift_periods)
    
    # Set first 6 values to NaN since we don't have previous day's data
    prev_high[:shift_periods] = np.nan
    prev_low[:shift_periods] = np.nan
    prev_close[:shift_periods] = np.nan
    
    # Calculate Camarilla levels
    high_low_range = prev_high - prev_low
    camarilla_r3 = prev_close + high_low_range * 1.1 / 2
    camarilla_s3 = prev_close - high_low_range * 1.1 / 2
    camarilla_r1 = prev_close + high_low_range * 1.1 / 12
    camarilla_s1 = prev_close - high_low_range * 1.1 / 12
    
    # EMA34 for 1-day trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_rising = np.zeros_like(ema_34_1d, dtype=bool)
    ema_34_falling = np.zeros_like(ema_34_1d, dtype=bool)
    ema_34_rising[1:] = ema_34_1d[1:] > ema_34_1d[:-1]
    ema_34_falling[1:] = ema_34_1d[1:] < ema_34_1d[:-1]
    
    ema_34_rising_aligned = align_htf_to_ltf(prices, df_1d, ema_34_rising)
    ema_34_falling_aligned = align_htf_to_ltf(prices, df_1d, ema_34_falling)
    
    # Volume confirmation: current volume > 2.0 * 20-period EMA of volume
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, shift_periods) + 5  # Ensure we have enough data for all indicators
    
    for i in range(start_idx, n):
        if (np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or np.isnan(camarilla_r1[i]) or 
            np.isnan(camarilla_s1[i]) or np.isnan(ema_34_rising_aligned[i]) or np.isnan(ema_34_falling_aligned[i]) or 
            np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Close > Camarilla R3 AND EMA34(1d) rising AND volume spike
            long_condition = (close[i] > camarilla_r3[i]) and ema_34_rising_aligned[i] and volume_spike[i]
            # Short: Close < Camarilla S3 AND EMA34(1d) falling AND volume spike
            short_condition = (close[i] < camarilla_s3[i]) and ema_34_falling_aligned[i] and volume_spike[i]
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Close < Camarilla R1
            if close[i] < camarilla_r1[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Close > Camarilla S1
            if close[i] > camarilla_s1[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals