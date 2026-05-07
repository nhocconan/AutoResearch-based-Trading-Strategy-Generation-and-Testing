#!/usr/bin/env python3
name = "12h_Camarilla_R3_S3_Breakout_1dTrend_Volume"
timeframe = "12h"
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
    
    # 1d EMA34 for trend filter
    close_1d = pd.Series(df_1d['close'])
    ema34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate 1d Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_vals = df_1d['close'].values
    
    # Camarilla formula: R3 = close + (high - low) * 1.1/4, S3 = close - (high - low) * 1.1/4
    camarilla_range = (high_1d - low_1d) * 1.1
    r3 = close_1d_vals + camarilla_range / 4
    s3 = close_1d_vals - camarilla_range / 4
    
    # Align Camarilla levels to 12h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # 12h volume spike: > 1.5x 24-period average (12 days)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_spike = volume > 1.5 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for volume MA and warmup
    
    for i in range(start_idx, n):
        if np.isnan(ema34_1d_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above R3, 1d trend up (price > EMA34), volume spike
            if close[i] > r3_aligned[i] and close[i] > ema34_1d_aligned[i] and vol_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S3, 1d trend down (price < EMA34), volume spike
            elif close[i] < s3_aligned[i] and close[i] < ema34_1d_aligned[i] and vol_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Price falls below S3 or 1d trend turns down
            if close[i] < s3_aligned[i] or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price rises above R3 or 1d trend turns up
            if close[i] > r3_aligned[i] or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation.
# Long when price breaks above R3 (Camarilla resistance), 1d trend is up (price > EMA34), and volume spike confirms.
# Short when price breaks below S3 (Camarilla support), 1d trend is down (price < EMA34), and volume spike confirms.
# Uses 1d timeframe for trend and Camarilla levels to avoid whipsaws, 12h for entry timing.
# Volume spike (>1.5x 24-period average) ensures conviction. Discrete 0.25 position size limits risk.
# Works in bull markets (trend + breakout) and bear markets (reverse criteria).
# Target: 15-30 trades/year to minimize fee drag while capturing sustained moves.