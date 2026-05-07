#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-day Camarilla Pivot (S3/R3) breakout with weekly trend filter (EMA50) and volume confirmation.
# Long when: Close > R3 AND weekly EMA50 rising AND volume > 1.5 * EMA20(volume).
# Short when: Close < S3 AND weekly EMA50 falling AND volume > 1.5 * EMA20(volume).
# Exit when price crosses back to daily pivot point (PP).
# Designed for low trade frequency (target: 10-25/year) to minimize fee drift and improve generalization.
# Works in bull markets via upward breakouts and in bear markets via downward breakouts.
# Uses 1d primary timeframe and 1h weekly trend filter as per experiment #136464.
name = "1d_Camarilla_R3S3_1wEMA50_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Camarilla Pivot Levels from previous day
    # PP = (H + L + C) / 3
    # R3 = C + (H - L) * 1.1
    # S3 = C - (H - L) * 1.1
    # We use previous day's high, low, close to avoid look-ahead
    pp = (np.roll(high, 1) + np.roll(low, 1) + np.roll(close, 1)) / 3
    r3 = np.roll(close, 1) + (np.roll(high, 1) - np.roll(low, 1)) * 1.1
    s3 = np.roll(close, 1) - (np.roll(high, 1) - np.roll(low, 1)) * 1.1
    
    # Set first value to NaN since we don't have previous day
    pp[0] = np.nan
    r3[0] = np.nan
    s3[0] = np.nan
    
    # EMA10 for exit (using daily close)
    ema_10 = pd.Series(close).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Load weekly data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # EMA50 on weekly close
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Rising if current > previous, falling if current < previous
    ema_50_rising = np.zeros_like(ema_50_1w, dtype=bool)
    ema_50_falling = np.zeros_like(ema_50_1w, dtype=bool)
    ema_50_rising[1:] = ema_50_1w[1:] > ema_50_1w[:-1]
    ema_50_falling[1:] = ema_50_1w[1:] < ema_50_1w[:-1]
    
    ema_50_rising_aligned = align_htf_to_ltf(prices, df_1w, ema_50_rising)
    ema_50_falling_aligned = align_htf_to_ltf(prices, df_1w, ema_50_falling)
    
    # Volume confirmation: current volume > 1.5 * 20-period EMA of volume
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(pp[i]) or np.isnan(r3[i]) or np.isnan(s3[i]) or np.isnan(ema_10[i]) or 
            np.isnan(ema_50_rising_aligned[i]) or np.isnan(ema_50_falling_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Close > R3 AND weekly EMA50 rising AND volume spike
            long_condition = (close[i] > r3[i]) and ema_50_rising_aligned[i] and volume_spike[i]
            # Short: Close < S3 AND weekly EMA50 falling AND volume spike
            short_condition = (close[i] < s3[i]) and ema_50_falling_aligned[i] and volume_spike[i]
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Close < PP (pivot point)
            if close[i] < pp[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Close > PP (pivot point)
            if close[i] > pp[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals