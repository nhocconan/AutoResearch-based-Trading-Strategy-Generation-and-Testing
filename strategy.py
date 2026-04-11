#!/usr/bin/env python3
# 12h_1d_camarilla_pivot_volume_v1
# Strategy: 12h Camarilla pivot bounce with volume confirmation and 1d trend filter
# Timeframe: 12h
# Leverage: 1.0
# Hypothesis: Camarilla pivot levels (S3/S4 for long, R3/R4 for short) act as strong support/resistance.
# Long when price bounces from S3/S4 with volume confirmation and 1d close > EMA50 (uptrend).
# Short when price reverses from R3/R4 with volume confirmation and 1d close < EMA50 (downtrend).
# Uses 1d EMA50 for trend filter to avoid counter-trend trades. Designed for low trade frequency
# (~15-30/year) to minimize fee drag. Works in bull markets via S3/S4 bounces in uptrend and
# bear markets via R3/R4 reversals in downtrend.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_camarilla_pivot_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Previous 1d high, low, close for Camarilla calculation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1d bar
    # H = high, L = low, C = close
    # Resistance levels: R4 = C + (H-L)*1.5/2, R3 = C + (H-L)*1.25/2, R2 = C + (H-L)*1.1/2, R1 = C + (H-L)*0.5/2
    # Support levels: S1 = C - (H-L)*0.5/2, S2 = C - (H-L)*1.1/2, S3 = C - (H-L)*1.25/2, S4 = C - (H-L)*1.5/2
    H_L = high_1d - low_1d
    C = close_1d
    
    R4 = C + H_L * 1.5 / 2
    R3 = C + H_L * 1.25 / 2
    S3 = C - H_L * 1.25 / 2
    S4 = C - H_L * 1.5 / 2
    
    # Align Camarilla levels to 12h timeframe
    R4_aligned = align_htf_to_ltf(prices, df_1d, R4)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    S4_aligned = align_htf_to_ltf(prices, df_1d, S4)
    
    # 12h volume average (20-period) for confirmation
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if np.isnan(ema_50_1d_aligned[i]) or np.isnan(R4_aligned[i]) or np.isnan(R3_aligned[i]) or \
           np.isnan(S3_aligned[i]) or np.isnan(S4_aligned[i]) or np.isnan(vol_avg_20[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume[i] > 1.5 * vol_avg_20[i]
        
        # Trend filter: 1d EMA50 slope (using current vs previous)
        trending_up = ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1]
        trending_down = ema_50_1d_aligned[i] < ema_50_1d_aligned[i-1]
        
        # Entry conditions
        # Long: price touches/bounces from S3 or S4 AND volume confirmation AND 1d uptrend
        if ((low[i] <= S3_aligned[i] * 1.001 and low[i] >= S3_aligned[i] * 0.999) or
            (low[i] <= S4_aligned[i] * 1.001 and low[i] >= S4_aligned[i] * 0.999)) and \
           vol_confirm and trending_up and position != 1:
            position = 1
            signals[i] = 0.25
        # Short: price touches/reverses from R3 or R4 AND volume confirmation AND 1d downtrend
        elif ((high[i] >= R3_aligned[i] * 0.999 and high[i] <= R3_aligned[i] * 1.001) or
              (high[i] >= R4_aligned[i] * 0.999 and high[i] <= R4_aligned[i] * 1.001)) and \
             vol_confirm and trending_down and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: opposite touch (price reaches opposite level) or trend change
        elif position == 1 and ((high[i] >= R3_aligned[i] * 0.999 and high[i] <= R3_aligned[i] * 1.001) or
                                (high[i] >= R4_aligned[i] * 0.999 and high[i] <= R4_aligned[i] * 1.001)):
            position = 0
            signals[i] = 0.0
        elif position == -1 and ((low[i] <= S3_aligned[i] * 1.001 and low[i] >= S3_aligned[i] * 0.999) or
                                 (low[i] <= S4_aligned[i] * 1.001 and low[i] >= S4_aligned[i] * 0.999)):
            position = 0
            signals[i] = 0.0
        elif position == 1 and not trending_up:
            position = 0
            signals[i] = 0.0
        elif position == -1 and not trending_down:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals