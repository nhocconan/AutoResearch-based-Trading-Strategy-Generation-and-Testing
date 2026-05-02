#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R4/S4 breakout with 1d EMA50 trend filter and volume confirmation
# Uses 12h primary timeframe for signal generation with tighter Camarilla R4/S4 breakouts
# 1d EMA50 trend filter provides higher timeframe bias for stronger trend alignment
# Volume confirmation (2.5x 20-period average) ensures institutional participation
# Discrete position sizing (0.25) controls fee drag while maintaining profit potential
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe
# Works in both bull and bear markets by only trading in direction of 1d trend
# R4/S4 levels offer stronger breakout validation than R3/S3, reducing false signals

name = "12h_Camarilla_R4S4_Breakout_1dEMA50_Trend_Volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate 12h Camarilla levels (based on previous 12h bar's OHLC)
    # Camarilla: R4 = C + (H-L)*1.1/2, S4 = C - (H-L)*1.1/2
    # where C = close, H = high, L = low of previous period
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close[0] = np.nan
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    
    camarilla_r4 = prev_close + (prev_high - prev_low) * 1.1 / 2
    camarilla_s4 = prev_close - (prev_high - prev_low) * 1.1 / 2
    
    # Volume confirmation (2.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for indicators)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(camarilla_r4[i]) or 
            np.isnan(camarilla_s4[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Break above Camarilla R4 + volume spike + price > 1d EMA50
            if close[i] > camarilla_r4[i] and volume_spike[i] and close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Break below Camarilla S4 + volume spike + price < 1d EMA50
            elif close[i] < camarilla_s4[i] and volume_spike[i] and close[i] < ema50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Close below Camarilla S4 or price < 1d EMA50
            if close[i] < camarilla_s4[i] or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Close above Camarilla R4 or price > 1d EMA50
            if close[i] > camarilla_r4[i] or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals