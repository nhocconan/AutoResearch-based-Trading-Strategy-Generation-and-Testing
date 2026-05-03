#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla H1/L1 breakout with 1w EMA(50) trend filter and volume confirmation
# Uses weekly EMA for stronger trend alignment (less whipsaw than 1d) and Camarilla H1/L1 from prior weekly bar.
# Volume confirmation (2.0x 20-period average) ensures institutional participation.
# Designed for low trade frequency (12-37/year) to minimize fee drag. Works in both bull and bear markets.
# H1/L1 levels are tighter than R3/S3, providing more precise entries while maintaining structure.

name = "12h_Camarilla_H1L1_Breakout_1wEMA50_Trend_Volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA(50) trend filter and Camarilla calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate EMA(50) on 1w for trend filter
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w EMA to 12h timeframe (wait for completed 1w bar)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Camarilla levels from prior 1w bar: H1, L1
    # Camarilla: H1 = close + 1.1*(high-low)/2, L1 = close - 1.1*(high-low)/2
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    camarilla_h1 = close_1w + 1.1 * (high_1w - low_1w) / 2
    camarilla_l1 = close_1w - 1.1 * (high_1w - low_1w) / 2
    
    # Align Camarilla levels to 12h timeframe (wait for completed 1w bar)
    camarilla_h1_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h1)
    camarilla_l1_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l1)
    
    # Volume confirmation (2.0x 20-period average) on 12h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for all calculations)
    start_idx = 70  # max(50 for 1w EMA, 20 for volume MA +1 for shift)
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(camarilla_h1_aligned[i]) or 
            np.isnan(camarilla_l1_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Close above Camarilla H1 + price above 1w EMA(50) + volume spike
            if (close[i] > camarilla_h1_aligned[i] and close[i] > ema_50_1w_aligned[i] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: Close below Camarilla L1 + price below 1w EMA(50) + volume spike
            elif (close[i] < camarilla_l1_aligned[i] and close[i] < ema_50_1w_aligned[i] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Close below Camarilla L1 (reversal) or price below 1w EMA(50) (trend reversal)
            if close[i] < camarilla_l1_aligned[i] or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Close above Camarilla H1 (reversal) or price above 1w EMA(50) (trend reversal)
            if close[i] > camarilla_h1_aligned[i] or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals