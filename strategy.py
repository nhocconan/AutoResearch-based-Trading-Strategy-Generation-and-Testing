#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R1/S1 breakout with 12h EMA50 trend filter and volume spike confirmation
# Camarilla pivot levels provide intraday support/resistance. Breakout above R1 or below S1 with
# volume spike indicates institutional participation. 12h EMA50 ensures alignment with medium-term trend.
# Designed to work in both bull and bear markets by taking breakouts in direction of 12h trend.
# Target: 20-50 trades/year to minimize fee drag while maintaining edge.

name = "4h_Camarilla_R1_S1_Breakout_12hEMA50_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h HTF data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # 1d data for Camarilla pivot calculation (yesterday's OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1d bar
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Camarilla levels: R4, R3, R2, R1, PP, S1, S2, S3, S4
    # We only need R1 and S1 for breakout signals
    # R1 = Close + 1.1*(High-Low)/12
    # S1 = Close - 1.1*(High-Low)/12
    camarilla_range = high_1d - low_1d
    r1 = close_1d + 1.1 * camarilla_range / 12
    s1 = close_1d - 1.1 * camarilla_range / 12
    
    # Align Camarilla levels to 4h timeframe (use previous day's levels)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Volume confirmation: current volume > 2.0 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = 50  # Need sufficient history for EMA50 and Camarilla
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 12h EMA50
        uptrend = close[i] > ema_50_12h_aligned[i]
        downtrend = close[i] < ema_50_12h_aligned[i]
        
        # Volume confirmation
        vol_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: break above R1, volume spike, uptrend
            if close[i] > r1_aligned[i] and vol_spike and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: break below S1, volume spike, downtrend
            elif close[i] < s1_aligned[i] and vol_spike and downtrend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on break below R1 or trend reversal
            if close[i] < r1_aligned[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on break above S1 or trend reversal
            if close[i] > s1_aligned[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals