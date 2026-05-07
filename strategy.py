#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Camarilla Pivot reversal with 12-hour trend filter and volume confirmation.
# Long when: Price touches or crosses below S3 level AND 12h EMA50 rising AND volume spike.
# Short when: Price touches or crosses above R3 level AND 12h EMA50 falling AND volume spike.
# Exit when price crosses back above/below the 20-period EMA.
# Uses Camarilla pivot levels from daily timeframe for institutional-level support/resistance.
# Designed for low trade frequency (target: 20-35/year) to minimize fee drag and improve generalization.
# Works in both bull and bear markets by fading extreme moves at key pivot levels.
name = "4h_Camarilla_S3R3_12hEMA50_Volume"
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
    
    # Calculate daily Camarilla pivot levels (using previous day's OHLC)
    # Camarilla formula: 
    # H4 = Close + 1.1 * (High - Low) / 2
    # L4 = Close - 1.1 * (High - Low) / 2
    # H3 = Close + 1.1 * (High - Low) / 4
    # L3 = Close - 1.1 * (High - Low) / 4
    # H2 = Close + 1.1 * (High - Low) / 6
    # L2 = Close - 1.1 * (High - Low) / 6
    # H1 = Close + 1.1 * (High - Low) / 12
    # L1 = Close - 1.1 * (High - Low) / 12
    # Pivot = (High + Low + Close) / 3
    # We'll use S3 (L3) and R3 (H3) as our entry levels
    
    # Get daily data for pivot calculation
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 2:
        return np.zeros(n)
    
    daily_high = df_daily['high'].values
    daily_low = df_daily['low'].values
    daily_close = df_daily['close'].values
    
    # Calculate Camarilla levels for each day
    camarilla_S3 = np.zeros_like(daily_close)
    camarilla_R3 = np.zeros_like(daily_close)
    
    for i in range(1, len(daily_close)):
        high_low = daily_high[i-1] - daily_low[i-1]
        camarilla_S3[i] = daily_close[i-1] - 1.1 * high_low / 4
        camarilla_R3[i] = daily_close[i-1] + 1.1 * high_low / 4
    
    # Align daily Camarilla levels to 4h timeframe
    camarilla_S3_aligned = align_htf_to_ltf(prices, df_daily, camarilla_S3)
    camarilla_R3_aligned = align_htf_to_ltf(prices, df_daily, camarilla_R3)
    
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
    
    # Volume confirmation: current volume > 1.5 * 20-period EMA of volume
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(camarilla_S3_aligned[i]) or np.isnan(camarilla_R3_aligned[i]) or np.isnan(ema_20[i]) or 
            np.isnan(ema_50_rising_aligned[i]) or np.isnan(ema_50_falling_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price touches or crosses below S3 AND EMA50(12h) rising AND volume spike
            long_condition = (close[i] <= camarilla_S3_aligned[i]) and ema_50_rising_aligned[i] and volume_spike[i]
            # Short: Price touches or crosses above R3 AND EMA50(12h) falling AND volume spike
            short_condition = (close[i] >= camarilla_R3_aligned[i]) and ema_50_falling_aligned[i] and volume_spike[i]
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Cross above EMA20
            if close[i] > ema_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Cross below EMA20
            if close[i] < ema_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals