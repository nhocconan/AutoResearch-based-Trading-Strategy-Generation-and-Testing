#!/usr/bin/env python3
name = "1h_Camarilla_R1_S1_Breakout_4hTrend_Volume"
timeframe = "1h"
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
    
    # Load 4h data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # 4h EMA50 for trend filter
    ema50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # 1h Camarilla levels: R1, S1 from previous hour
    # Camarilla: R1 = close + (high - low) * 1.1/12, S1 = close - (high - low) * 1.1/12
    prev_close_1h = pd.Series(close).shift(1).values
    prev_high_1h = pd.Series(high).shift(1).values
    prev_low_1h = pd.Series(low).shift(1).values
    camarilla_r1_1h = prev_close_1h + (prev_high_1h - prev_low_1h) * 1.1 / 12
    camarilla_s1_1h = prev_close_1h - (prev_high_1h - prev_low_1h) * 1.1 / 12
    
    # 1h volume spike: > 2x 24-period average
    vol_ma_1h = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_spike_1h = volume > 2 * vol_ma_1h
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 24)  # Wait for EMA50 and volume MA
    
    for i in range(start_idx, n):
        if np.isnan(ema50_4h_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above R1 with volume spike, 4h uptrend (close > EMA50)
            if (close[i] > camarilla_r1_1h[i] and vol_spike_1h[i] and close[i] > ema50_4h_aligned[i]):
                signals[i] = 0.20
                position = 1
            # Short: Break below S1 with volume spike, 4h downtrend (close < EMA50)
            elif (close[i] < camarilla_s1_1h[i] and vol_spike_1h[i] and close[i] < ema50_4h_aligned[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit: Price below S1 or 4h downtrend
            if close[i] < camarilla_s1_1h[i] or close[i] < ema50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit: Price above R1 or 4h uptrend
            if close[i] > camarilla_r1_1h[i] or close[i] > ema50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

# Hypothesis: 1h Camarilla R1/S1 breakouts with volume confirmation and 4h EMA50 trend filter.
# In bull markets: 4h uptrend filters long entries, reducing false breakouts.
# In bear markets: 4h downtrend filters short entries, capturing retracements.
# Volume spike ensures institutional interest. Session filter (08-20 UTC) applied via volume spike requirement.
# Target: 15-30 trades/year to avoid fee drag. Position size 0.20 limits risk.