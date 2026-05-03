#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA(50) trend filter and volume confirmation
# Camarilla pivot levels (R3/S3) derived from prior 4h range act as strong support/resistance.
# Breakout above R3 or below S3 with volume confirmation indicates institutional participation.
# 4h EMA(50) ensures we trade with the higher timeframe trend to avoid counter-trend whipsaws.
# Session filter (08-20 UTC) reduces noise trades. Designed for 15-37 trades/year on 1h timeframe.

name = "1h_Camarilla_R3S3_Breakout_4hEMA50_Trend_Volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    
    # Get 4h data for EMA(50) trend filter and Camarilla calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Calculate EMA(50) on 4h for trend filter
    ema_50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 4h EMA to 1h timeframe (wait for completed 4h bar)
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate Camarilla levels from prior 4h bar: R3, S3
    # Camarilla: R3 = close + 1.1*(high-low)/4, S3 = close - 1.1*(high-low)/4
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    camarilla_r3 = close_4h + 1.1 * (high_4h - low_4h) / 4
    camarilla_s3 = close_4h - 1.1 * (high_4h - low_4h) / 4
    
    # Align Camarilla levels to 1h timeframe (wait for completed 4h bar)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s3)
    
    # Volume confirmation (2.0x 20-period average) on 1h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for all calculations)
    start_idx = 51  # max(50 for 4h EMA, 20 for volume MA +1 for shift)
    
    for i in range(start_idx, n):
        # Session filter: 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # Check for NaN values in indicators
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Close above Camarilla R3 + price above 4h EMA(50) + volume spike
            if (close[i] > camarilla_r3_aligned[i] and close[i] > ema_50_4h_aligned[i] and volume_spike[i]):
                signals[i] = 0.20
                position = 1
            # Short entry: Close below Camarilla S3 + price below 4h EMA(50) + volume spike
            elif (close[i] < camarilla_s3_aligned[i] and close[i] < ema_50_4h_aligned[i] and volume_spike[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Close below Camarilla S3 (reversal) or price below 4h EMA(50) (trend reversal)
            if close[i] < camarilla_s3_aligned[i] or close[i] < ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit: Close above Camarilla R3 (reversal) or price above 4h EMA(50) (trend reversal)
            if close[i] > camarilla_r3_aligned[i] or close[i] > ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals