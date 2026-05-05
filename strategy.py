#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla pivot breakout + 1w EMA trend filter + volume confirmation
# Uses 1d timeframe with 1w HTF for trend alignment to minimize trades and fee drag
# Long when: price breaks above Camarilla R3 AND 1w EMA50 is rising AND volume > 1.5x 20-period MA
# Short when: price breaks below Camarilla S3 AND 1w EMA50 is falling AND volume > 1.5x 20-period MA
# Exit when: price returns to Camarilla Pivot level (mean reversion to equilibrium)
# Timeframe: 1d, HTF: 1w. Target: 30-100 total trades over 4 years (7-25/year) to avoid fee drag.

name = "1d_Camarilla_R3S3_Breakout_1wEMA50_VolumeConfirm"
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
    
    # Calculate 1d Camarilla levels (based on previous day)
    # Pivot = (H + L + C) / 3
    # R3 = Pivot + 1.1 * (H - L)
    # S3 = Pivot - 1.1 * (H - L)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    camarilla_r3 = pivot + 1.1 * (prev_high - prev_low)
    camarilla_s3 = pivot - 1.1 * (prev_high - prev_low)
    
    # Get 1w data ONCE before loop for EMA calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:  # need sufficient data for EMA
        return np.zeros(n)
    
    # Calculate EMA(50) on 1w
    close_1w = df_1w['close'].values
    if len(close_1w) >= 50:
        ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
        # Rising/falling EMA: compare current to previous
        ema_rising = np.zeros(len(ema_50_1w), dtype=bool)
        ema_falling = np.zeros(len(ema_50_1w), dtype=bool)
        for i in range(1, len(ema_50_1w)):
            if not np.isnan(ema_50_1w[i]) and not np.isnan(ema_50_1w[i-1]):
                ema_rising[i] = ema_50_1w[i] > ema_50_1w[i-1]
                ema_falling[i] = ema_50_1w[i] < ema_50_1w[i-1]
    else:
        ema_50_1w = np.full(len(close_1w), np.nan)
        ema_rising = np.zeros(len(close_1w), dtype=bool)
        ema_falling = np.zeros(len(close_1w), dtype=bool)
    
    # Align 1w EMA indicators to 1d timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    ema_rising_aligned = align_htf_to_ltf(prices, df_1w, ema_rising.astype(float))
    ema_falling_aligned = align_htf_to_ltf(prices, df_1w, ema_falling.astype(float))
    
    # Volume confirmation on 1d
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(pivot[i]) or np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(ema_rising_aligned[i]) or np.isnan(ema_falling_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above R3 + rising EMA + volume filter
            if (close[i] > camarilla_r3[i] and 
                ema_rising_aligned[i] == 1.0 and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below S3 + falling EMA + volume filter
            elif (close[i] < camarilla_s3[i] and 
                  ema_falling_aligned[i] == 1.0 and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to pivot (mean reversion)
            if close[i] <= pivot[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to pivot (mean reversion)
            if close[i] >= pivot[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals