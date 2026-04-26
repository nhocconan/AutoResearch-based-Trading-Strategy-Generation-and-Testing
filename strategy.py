#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_1dEMA34_VolumeSpike_Dyn
Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation. 
In bull/bear markets, price tends to respect Camarilla pivot levels (R3/S3) as significant support/resistance.
Breakouts above R3 or below S3 with volume confirmation and aligned 1d trend capture strong moves.
Uses discrete position sizing (0.30) to minimize fee churn and targets 20-50 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for HTF trend and volume average
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate previous day's Camarilla levels (using prior 1d bar)
    # Camarilla: R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low)
    #          S3 = close - 1.1*(high-low), S4 = close - 1.5*(high-low)
    # We use the prior completed 1d bar to avoid look-ahead
    prev_close_1d = df_1d['close'].shift(1).values
    prev_high_1d = df_1d['high'].shift(1).values
    prev_low_1d = df_1d['low'].shift(1).values
    prev_range_1d = prev_high_1d - prev_low_1d
    
    # Calculate Camarilla levels for prior 1d bar
    R3_1d = prev_close_1d + 1.1 * prev_range_1d
    S3_1d = prev_close_1d - 1.1 * prev_range_1d
    
    # Align to 4h timeframe (these levels stay constant throughout the 4h bars of the day)
    R3_1d_aligned = align_htf_to_ltf(prices, df_1d, R3_1d)
    S3_1d_aligned = align_htf_to_ltf(prices, df_1d, S3_1d)
    
    # Calculate 1d EMA34 for trend filter (using prior completed 1d bar)
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().shift(1).values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    htf_trend = np.where(close > ema_34_1d_aligned, 1, -1)  # 1 = uptrend, -1 = downtrend
    
    # Volume spike: current 4h volume > 2.0 * average of last 24 4h bars (6h MA)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 34 for EMA, 24 for volume MA)
    start_idx = max(34, 24)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(R3_1d_aligned[i]) or np.isnan(S3_1d_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.30
            else:
                signals[i] = -0.30
            continue
        
        # Long breakout: price > R3 with volume spike and uptrend HTF
        if close[i] > R3_1d_aligned[i] and volume_spike[i] and htf_trend[i] == 1:
            if position != 1:
                signals[i] = 0.30
                position = 1
            else:
                signals[i] = 0.30
        
        # Short breakout: price < S3 with volume spike and downtrend HTF
        elif close[i] < S3_1d_aligned[i] and volume_spike[i] and htf_trend[i] == -1:
            if position != -1:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = -0.30
        
        # Exit conditions: reverse signal or loss of momentum
        elif position == 1 and (close[i] < S3_1d_aligned[i] or htf_trend[i] == -1):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (close[i] > R3_1d_aligned[i] or htf_trend[i] == 1):
            signals[i] = 0.0
            position = 0
        
        # Hold current position
        else:
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.30
            else:
                signals[i] = -0.30
    
    return signals

name = "4h_Camarilla_R3_S3_Breakout_1dEMA34_VolumeSpike_Dyn"
timeframe = "4h"
leverage = 1.0