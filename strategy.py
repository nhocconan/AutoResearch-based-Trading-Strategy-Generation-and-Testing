#!/usr/bin/env python3
name = "4h_Camarilla_R3S3_Breakout_1dTrend_Volume_Spike"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # 1d RSI for momentum filter
    delta = pd.Series(df_1d['close']).diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=14).mean()
    rs = gain / loss
    rsi1d = 100 - (100 / (1 + rs))
    rsi1d = rsi1d.fillna(50).values  # Neutral when undefined
    rsi1d_aligned = align_htf_to_ltf(prices, df_1d, rsi1d)
    
    # Calculate daily Camarilla pivot levels (R3, S3)
    # Pivot = (H + L + C) / 3
    pivot = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    # R3 = Close + (High - Low) * 1.1/2
    r3 = df_1d['close'] + (df_1d['high'] - df_1d['low']) * 1.1 / 2
    # S3 = Close - (High - Low) * 1.1/2
    s3 = df_1d['close'] - (df_1d['high'] - df_1d['low']) * 1.1 / 2
    
    pivot_vals = pivot.values
    r3_vals = r3.values
    s3_vals = s3.values
    
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_vals)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_vals)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_vals)
    
    # 4h volume spike: > 1.5x 24-period average (24*4h = 4 days)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_spike = volume > 1.5 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(24, 34)  # Wait for volume MA and EMA34
    
    for i in range(start_idx, n):
        if np.isnan(ema34_1d_aligned[i]) or np.isnan(rsi1d_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(pivot_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price above R3, EMA34 up, RSI > 50, volume spike
            if close[i] > r3_aligned[i] and ema34_1d_aligned[i] > ema34_1d_aligned[i-1] and rsi1d_aligned[i] > 50 and vol_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price below S3, EMA34 down, RSI < 50, volume spike
            elif close[i] < s3_aligned[i] and ema34_1d_aligned[i] < ema34_1d_aligned[i-1] and rsi1d_aligned[i] < 50 and vol_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Price below S3 or RSI < 40
            if close[i] < s3_aligned[i] or rsi1d_aligned[i] < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price above R3 or RSI > 60
            if close[i] > r3_aligned[i] or rsi1d_aligned[i] > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter, 1d RSI momentum filter, and volume spike confirmation.
# Long when price breaks above daily R3, 1d EMA34 rising, RSI > 50 (bullish momentum), and volume spike confirms.
# Short when price breaks below daily S3, 1d EMA34 falling, RSI < 50 (bearish momentum), and volume spike confirms.
# Uses 1d timeframe for trend/momentum/Camarilla levels to avoid whipsaws, 4h for entry timing.
# Volume spike (>1.5x average) ensures conviction. Discrete 0.25 position size limits risk.
# Works in bull markets (trend + momentum) and bear markets (reverse criteria).
# Target: 25-40 trades/year to minimize fee drag while capturing sustained moves.