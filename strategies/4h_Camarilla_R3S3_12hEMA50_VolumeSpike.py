#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Camarilla pivot breakout with 12-hour trend filter (EMA50) and volume spike.
# Long when: Close > Camarilla R3 AND EMA50(12h) rising AND volume > 2.0 * EMA20(volume).
# Short when: Close < Camarilla S3 AND EMA50(12h) falling AND volume > 2.0 * EMA20(volume).
# Exit when price crosses back below/above the 20-period EMA.
# Camarilla levels provide institutional support/resistance; EMA50 filters trend; volume spike confirms breakout.
# Designed for low trade frequency (target: 20-40/year) to minimize fee drift and improve generalization.
# Works in bull markets via upward breakouts at R3/R4 and in bear markets via downward breakouts at S3/S4.
name = "4h_Camarilla_R3S3_12hEMA50_VolumeSpike"
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
    
    # Camarilla levels: based on previous day's range
    # R3 = close + 1.1 * (high - low) / 2
    # S3 = close - 1.1 * (high - low) / 2
    # We use the previous day's high/low, so we need to shift by 1 day
    # Since we're on 4h timeframe, 1 day = 6 bars
    prev_day_high = np.roll(high, 6)
    prev_day_low = np.roll(low, 6)
    prev_day_close = np.roll(close, 6)
    # Set first 6 values to NaN since we don't have previous day data
    prev_day_high[:6] = np.nan
    prev_day_low[:6] = np.nan
    prev_day_close[:6] = np.nan
    
    camarilla_r3 = prev_day_close + 1.1 * (prev_day_high - prev_day_low) / 2
    camarilla_s3 = prev_day_close - 1.1 * (prev_day_high - prev_day_low) / 2
    
    # EMA20 for exit
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Load 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    # EMA50 on 12h close
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Rising if current > previous, falling if current < previous
    ema_50_rising = np.zeros_like(ema_50_12h, dtype=bool)
    ema_50_falling = np.zeros_like(ema_50_12h, dtype=bool)
    ema_50_rising[1:] = ema_50_12h[1:] > ema_50_12h[:-1]
    ema_50_falling[1:] = ema_50_12h[1:] < ema_50_12h[:-1]
    
    ema_50_rising_aligned = align_htf_to_ltf(prices, df_12h, ema_50_rising)
    ema_50_falling_aligned = align_htf_to_ltf(prices, df_12h, ema_50_falling)
    
    # Volume confirmation: current volume > 2.0 * 20-period EMA of volume
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or np.isnan(ema_20[i]) or 
            np.isnan(ema_50_rising_aligned[i]) or np.isnan(ema_50_falling_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Close > Camarilla R3 AND EMA50(12h) rising AND volume spike
            long_condition = (close[i] > camarilla_r3[i]) and ema_50_rising_aligned[i] and volume_spike[i]
            # Short: Close < Camarilla S3 AND EMA50(12h) falling AND volume spike
            short_condition = (close[i] < camarilla_s3[i]) and ema_50_falling_aligned[i] and volume_spike[i]
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Close < EMA20
            if close[i] < ema_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Close > EMA20
            if close[i] > ema_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals