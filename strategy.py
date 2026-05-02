#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA200 trend filter and volume spike confirmation
# Uses 4h EMA200 for higher timeframe trend filter to avoid counter-trend trades
# 1d Camarilla levels (R3/S3) provide precise support/resistance from prior day
# Volume confirmation (2.0x 20-period average) ensures institutional participation
# Session filter (08-20 UTC) reduces noise during low-liquidity hours
# Discrete position sizing (0.20) minimizes fee churn while maintaining exposure
# Target: 60-150 total trades over 4 years = 15-37/year for 1h timeframe
# Works in bull/bear markets: EMA200 filters trend direction, Camarilla breakouts capture momentum,
# volume confirmation reduces false signals, session filter avoids Asian session noise

name = "1h_Camarilla_R3S3_Breakout_4hEMA200_Trend_VolumeSpike_Session_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours (08-20 UTC) for filter
    hours = prices.index.hour
    
    # Load 1d data ONCE before loop for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1d bar
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    camarilla_r3 = close_1d + 1.1 * (high_1d - low_1d) / 2
    camarilla_s3 = close_1d - 1.1 * (high_1d - low_1d) / 2
    
    # Align Camarilla levels to 1h timeframe (use previous day's levels)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Load 4h data ONCE before loop for EMA200 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 200:
        return np.zeros(n)
    
    # Calculate 4h EMA200 for trend filter
    close_4h = df_4h['close'].values
    ema_200_4h = pd.Series(close_4h).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_200_4h)
    
    # Volume confirmation (2.0x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_confirm = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for indicators)
    start_idx = 200
    
    for i in range(start_idx, n):
        # Skip if outside trading session (08-20 UTC)
        if hours[i] < 8 or hours[i] > 20:
            signals[i] = 0.0
            continue
            
        # Check for NaN values in indicators
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema_200_4h_aligned[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Close > Camarilla R3 + price > 4h EMA200 + volume confirm + session
            if close[i] > camarilla_r3_aligned[i] and close[i] > ema_200_4h_aligned[i] and volume_confirm[i]:
                signals[i] = 0.20
                position = 1
            # Short: Close < Camarilla S3 + price < 4h EMA200 + volume confirm + session
            elif close[i] < camarilla_s3_aligned[i] and close[i] < ema_200_4h_aligned[i] and volume_confirm[i]:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Close < Camarilla S3 or price < 4h EMA200 (trend failure)
            if close[i] < camarilla_s3_aligned[i] or close[i] < ema_200_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit: Close > Camarilla R3 or price > 4h EMA200 (trend failure)
            if close[i] > camarilla_r3_aligned[i] or close[i] > ema_200_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals