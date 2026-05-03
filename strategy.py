#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1w EMA50 trend filter and volume confirmation
# Long when price breaks above Camarilla R3, price > 1w EMA50, and volume > 2.0x 20-bar average
# Short when price breaks below Camarilla S3, price < 1w EMA50, and volume > 2.0x 20-bar average
# Uses 1w EMA for higher timeframe trend alignment (matches experiment HTF)
# Volume spike confirms breakout strength
# Discrete position sizing (0.25) to minimize fee churn
# Designed for low trade frequency (12-37/year on 12h) to avoid fee drag
# Works in bull (breakouts above rising EMA) and bear (breakdowns below falling EMA)

name = "12h_Camarilla_R3S3_Breakout_1wEMA50_Trend_Volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA(50) trend filter and Camarilla calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate EMA(50) on 1w for trend filter
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w EMA to 12h timeframe (wait for completed 1w bar)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Camarilla levels on 12h from previous 1w OHLC
    # Camarilla: R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low), etc.
    # We use R3 and S3 for breakout
    # Since we need previous week's OHLC, we shift the 1w data by 1 bar
    df_1w_shifted = df_1w.copy()
    df_1w_shifted['open'] = df_1w_shifted['open'].shift(1)
    df_1w_shifted['high'] = df_1w_shifted['high'].shift(1)
    df_1w_shifted['low'] = df_1w_shifted['low'].shift(1)
    df_1w_shifted['close'] = df_1w_shifted['close'].shift(1)
    
    # Calculate Camarilla R3 and S3 from previous 1w
    # R3 = close + 1.1*(high-low)
    # S3 = close - 1.1*(high-low)
    camarilla_r3_1w = df_1w_shifted['close'] + 1.1 * (df_1w_shifted['high'] - df_1w_shifted['low'])
    camarilla_s3_1w = df_1w_shifted['close'] - 1.1 * (df_1w_shifted['high'] - df_1w_shifted['low'])
    
    # Align Camarilla levels to 12h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1w_shifted, camarilla_r3_1w.values)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1w_shifted, camarilla_s3_1w.values)
    
    # Volume confirmation (2.0x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for all calculations)
    start_idx = max(50, 20) + 1  # EMA(50) + volume MA(20) warmup + shift(1)
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price > Camarilla R3, price > 1w EMA50, volume spike
            if (close[i] > camarilla_r3_aligned[i] and 
                close[i] > ema_50_1w_aligned[i] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price < Camarilla S3, price < 1w EMA50, volume spike
            elif (close[i] < camarilla_s3_aligned[i] and 
                  close[i] < ema_50_1w_aligned[i] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price < Camarilla S3 or price < 1w EMA50
            if (close[i] < camarilla_s3_aligned[i] or 
                close[i] < ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price > Camarilla R3 or price > 1w EMA50
            if (close[i] > camarilla_r3_aligned[i] or 
                close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals