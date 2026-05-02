#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R4/S4 breakout with 1d EMA50 trend filter and volume spike confirmation
# R4/S4 levels represent stronger breakout points than R3/S3, reducing false breakouts.
# Combined with 1d EMA50 for higher timeframe trend alignment and volume spike for confirmation.
# Designed to work in both bull and bear markets via trend filter. Target: 20-40 trades/year.

name = "4h_Camarilla_R4S4_Breakout_1dEMA50_Trend_Volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate EMA(50) on 1d for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Camarilla pivot levels from previous day
    # R4 = Close + (High - Low) * 1.1/2
    # S4 = Close - (High - Low) * 1.1/2
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    diff = prev_high - prev_low
    r4 = prev_close + diff * 1.1 / 2
    s4 = prev_close - diff * 1.1 / 2
    
    # Align to 4h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Volume confirmation (2.0x 20-period average) on 4h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for calculations)
    start_idx = 20
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(r4_aligned[i]) or 
            np.isnan(s4_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Close breaks above R4 + 1d uptrend + volume spike
            if close[i] > r4_aligned[i] and close[i-1] <= r4_aligned[i-1] and close[i] > ema_50_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short entry: Close breaks below S4 + 1d downtrend + volume spike
            elif close[i] < s4_aligned[i] and close[i-1] >= s4_aligned[i-1] and close[i] < ema_50_1d_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Close breaks below R4 or trend reversal
            if close[i] < r4_aligned[i] and close[i-1] >= r4_aligned[i-1] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Close breaks above S4 or trend reversal
            if close[i] > s4_aligned[i] and close[i-1] <= s4_aligned[i-1] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals