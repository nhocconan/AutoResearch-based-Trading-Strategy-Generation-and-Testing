# 1. Hypothesis: 4h Camarilla R3/S3 Breakout with 1d EMA Trend Filter and Volume Spike Confirmation
#   - Camarilla pivot levels (R3/S3) act as strong support/resistance; breaks signal institutional participation.
#   - 1d EMA(34) ensures alignment with daily trend, reducing counter-trend trades.
#   - Volume spike (2.0x 20-period average) confirms breakout authenticity.
#   - Works in bull markets (buy R3 breaks in uptrend) and bear markets (sell S3 breaks in downtrend).
#   - Position size 0.25 targets ~25-35 trades/year, avoiding excessive fee drag.
#   - Exit when price returns to Camarilla pivot point (central equilibrium) or volume drops.

#!/usr/bin/env python3
name = "4h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike"
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
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA(34) for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from previous 1d bar (requires previous day's HLC)
    # We'll compute daily Camarilla and align to 4h
    # Camarilla formulas:
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # R4 = C + (Range * 1.5000)
    # R3 = C + (Range * 1.2500)
    # R2 = C + (Range * 1.1666)
    # R1 = C + (Range * 1.0833)
    # S1 = C - (Range * 1.0833)
    # S2 = C - (Range * 1.1666)
    # S3 = C - (Range * 1.2500)
    # S4 = C - (Range * 1.5000)
    
    # Use previous day's OHLC to avoid look-ahead
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Calculate Camarilla levels for each day (based on that day's OHLC)
    pivot = (daily_high + daily_low + daily_close) / 3
    rng = daily_high - daily_low
    r3 = daily_close + (rng * 1.2500)
    s3 = daily_close - (rng * 1.2500)
    pivot_level = pivot  # Central pivot for exit
    
    # Align Camarilla levels to 4h timeframe (using previous day's levels)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_level)
    
    # 4h volume spike (20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Wait for EMA and Camarilla
    
    for i in range(start_idx, n):
        if np.isnan(ema_34_1d_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or \
           np.isnan(pivot_aligned[i]) or np.isnan(vol_ma_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above R3 with volume and 1d uptrend
            vol_condition = volume[i] > vol_ma_20[i] * 2.0
            uptrend = ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]  # Rising EMA
            
            if close[i] > r3_aligned[i] and vol_condition and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: break below S3 with volume and 1d downtrend
            elif close[i] < s3_aligned[i] and vol_condition and not uptrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price returns to pivot point or volume drops significantly
            if close[i] <= pivot_aligned[i] or volume[i] < vol_ma_20[i] * 1.3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price returns to pivot point or volume drops significantly
            if close[i] >= pivot_aligned[i] or volume[i] < vol_ma_20[i] * 1.3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 4h Camarilla R3/S3 Breakout with 1d EMA Trend Filter and Volume Spike Confirmation
# - Camarilla R3/S3 levels represent strong intraday support/resistance; breaks indicate institutional flow.
# - 1d EMA(34) ensures trades align with daily trend, improving win rate in trending markets.
# - Volume spike (2.0x average) filters out false breakouts.
# - Works in bull (buy R3 breaks in uptrend) and bear (sell S3 breaks in downtrend).
# - Exit at daily pivot point provides logical mean-reversion target.
# - Position size 0.25 targets ~25-35 trades/year, balancing opportunity and fee efficiency.