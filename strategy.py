#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator with 1w trend filter and volume confirmation
# The Alligator (Jaw/Teeth/Lips) identifies trends via smoothed SMAs.
# In trending markets (JAW > TEETH > LIPS for up, JAW < TEETH < LIPS for down),
# trade pullbacks to the TEETH (middle line) with volume confirmation.
# Uses weekly timeframe to filter direction and daily for entry.
# Target: 10-25 trades/year to minimize fee drag while capturing trending moves.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d Williams Alligator ===
    # Jaw (13-period SMMA, offset 8), Teeth (8-period SMMA, offset 5), Lips (5-period SMMA, offset 3)
    # Using EMA as approximation for SMMA with same period
    jaw = pd.Series(close).ewm(span=13, adjust=False).mean().shift(8)
    teeth = pd.Series(close).ewm(span=8, adjust=False).mean().shift(5)
    lips = pd.Series(close).ewm(span=5, adjust=False).mean().shift(3)
    
    # === 1w Trend Filter (EMA 34) ===
    df_1w = get_htf_data(prices, '1w')
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # === 1d Volume Confirmation (vs 20-period average) ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 50
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Alligator alignment
        jaw_up = jaw[i] > teeth[i] and teeth[i] > lips[i]
        jaw_down = jaw[i] < teeth[i] and teeth[i] < lips[i]
        
        # Weekly trend filter
        weekly_up = close[i] > ema_34_1w_aligned[i]
        weekly_down = close[i] < ema_34_1w_aligned[i]
        
        # Volume confirmation
        vol_confirm = volume[i] > vol_ma_20[i] * 1.5
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: Alligator bullish + weekly up + volume
            if jaw_up and weekly_up and vol_confirm:
                signals[i] = 0.25
                position = 1
                continue
            # Short: Alligator bearish + weekly down + volume
            elif jaw_down and weekly_down and vol_confirm:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic: reverse Alligator alignment or weekly trend change
        elif position == 1:
            # Exit long if Alligator turns bearish OR weekly trend turns down
            if jaw_down or not weekly_up:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short if Alligator turns bullish OR weekly trend turns up
            if jaw_up or not weekly_down:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WilliamsAlligator_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0