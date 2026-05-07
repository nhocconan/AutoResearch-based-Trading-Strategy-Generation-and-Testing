#!/usr/bin/env python3
name = "4h_Camarilla_R1S1_Breakout_12hTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # 12h EMA50 for trend filter
    ema50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Calculate 12h volume spike: > 1.8x 24-period average (12 days)
    vol_12h = df_12h['volume'].values
    vol_ma_12h = pd.Series(vol_12h).rolling(window=24, min_periods=24).mean().values
    vol_spike_12h = vol_12h > 1.8 * vol_ma_12h
    vol_spike_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_spike_12h.astype(float))
    
    # Calculate Camarilla levels from previous 1d (using close only)
    # For 4h timeframe, we need daily data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Calculate Camarilla R1 and S1 levels
    camarilla_r1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    camarilla_s1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 24)  # Wait for EMA50 and volume MA
    
    for i in range(start_idx, n):
        if np.isnan(ema50_12h_aligned[i]) or np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above Camarilla R1 with 12h uptrend and volume spike
            if (close[i] > camarilla_r1_aligned[i] and 
                ema50_12h_aligned[i] > ema50_12h_aligned[max(i-1, start_idx)] and 
                vol_spike_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Camarilla S1 with 12h downtrend and volume spike
            elif (close[i] < camarilla_s1_aligned[i] and 
                  ema50_12h_aligned[i] < ema50_12h_aligned[max(i-1, start_idx)] and 
                  vol_spike_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Price breaks below Camarilla S1 or trend reverses
            if (close[i] < camarilla_s1_aligned[i] or 
                ema50_12h_aligned[i] < ema50_12h_aligned[max(i-1, start_idx)]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price breaks above Camarilla R1 or trend reverses
            if (close[i] > camarilla_r1_aligned[i] or 
                ema50_12h_aligned[i] > ema50_12h_aligned[max(i-1, start_idx)]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 4h Camarilla R1/S1 breakout with 12h EMA50 trend filter and 12h volume spike confirmation.
# Long when price breaks above Camarilla R1 level, 12h EMA50 is rising, and 12h volume spike occurs.
# Short when price breaks below Camarilla S1 level, 12h EMA50 is falling, and 12h volume spike occurs.
# Uses 12h timeframe for trend/volume confirmation to avoid whipsaws, 4h for entry timing.
# Camarilla levels provide precise intraday support/resistance based on previous day's range.
# Volume spike (>1.8x average) ensures institutional participation. Discrete 0.25 position size limits risk.
# Works in bull markets (breakouts with volume) and bear markets (breakdowns with volume).
# Target: 20-50 trades/year to minimize fee drag while capturing meaningful moves.