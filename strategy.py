#!/usr/bin/env python3
name = "4h_1d_Triple_Confirmation_Breakout"
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
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily ATR for volatility filter
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_10 = tr.rolling(window=10, min_periods=10).mean().values
    
    # Daily Donchian channels (20-period)
    donch_high_20 = df_1d['high'].rolling(window=20, min_periods=20).max().values
    donch_low_20 = df_1d['low'].rolling(window=20, min_periods=20).min().values
    
    # Daily EMA(50) for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align daily indicators to 4h timeframe
    donch_high_20_aligned = align_htf_to_ltf(prices, df_1d, donch_high_20)
    donch_low_20_aligned = align_htf_to_ltf(prices, df_1d, donch_low_20)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    atr_10_aligned = align_htf_to_ltf(prices, df_1d, atr_10)
    
    # 4h volume confirmation: volume spike detection
    vol_ma_10 = pd.Series(volume).rolling(window=10, min_periods=10).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20, 10)  # Wait for EMA50, Donchian20, VolMA10
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donch_high_20_aligned[i]) or np.isnan(donch_low_20_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(atr_10_aligned[i]) or
            np.isnan(vol_ma_10[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above Donchian high + above EMA50 + volume spike + ATR filter
            vol_condition = volume[i] > vol_ma_10[i] * 2.0
            above_ema = close[i] > ema_50_1d_aligned[i]
            atr_filter = atr_10_aligned[i] > 0  # Ensure volatility exists
            
            if (close[i] > donch_high_20_aligned[i] and 
                above_ema and 
                vol_condition and 
                atr_filter):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low + below EMA50 + volume spike + ATR filter
            elif (close[i] < donch_low_20_aligned[i] and 
                  not above_ema and 
                  vol_condition and 
                  atr_filter):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Price closes below Donchian low or volume drops significantly
            if close[i] < donch_low_20_aligned[i] or volume[i] < vol_ma_10[i] * 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price closes above Donchian high or volume drops significantly
            if close[i] > donch_high_20_aligned[i] or volume[i] < vol_ma_10[i] * 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 4h Donchian breakout with daily trend and volume confirmation
# - Uses daily Donchian channels (20-period) as dynamic support/resistance
# - Requires price to be above/below daily EMA(50) for trend alignment
# - Volume spike (2x 10-period average) confirms institutional participation
# - ATR filter ensures sufficient volatility for meaningful moves
# - Works in both bull (buy breakouts in uptrend) and bear (sell breakdowns in downtrend)
# - Position size 0.25 targets ~25-40 trades/year, minimizing fee drag
# - Simple 3-condition entry reduces overfitting and improves robustness
# - Exit on reverse Donchian break or volume collapse prevents whipsaws