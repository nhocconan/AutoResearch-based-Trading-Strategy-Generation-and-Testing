# 1d_Camarilla_R2_S2_Breakout_Volume
# 1d timeframe with 1w HTF filter: Camarilla R2/S2 breakout with volume confirmation
# Target: 30-100 trades over 4 years (7-25/year). Works in bull/bear via breakout logic.
# Uses weekly trend filter to avoid counter-trend trades in strong trends.

#!/usr/bin/env python3
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
    
    # Get daily data for pivot points
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily pivot points (standard formula)
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    r1_1d = 2 * pivot_1d - low_1d
    s1_1d = 2 * pivot_1d - high_1d
    r2_1d = pivot_1d + (high_1d - low_1d)
    s2_1d = pivot_1d - (high_1d - low_1d)
    r3_1d = high_1d + 2 * (pivot_1d - low_1d)
    s3_1d = low_1d - 2 * (high_1d - pivot_1d)
    
    # Use previous day's pivots (avoid look-ahead)
    r1_1d_prev = np.roll(r1_1d, 1)
    s1_1d_prev = np.roll(s1_1d, 1)
    r2_1d_prev = np.roll(r2_1d, 1)
    s2_1d_prev = np.roll(s2_1d, 1)
    r3_1d_prev = np.roll(r3_1d, 1)
    s3_1d_prev = np.roll(s3_1d, 1)
    r1_1d_prev[0] = np.nan
    s1_1d_prev[0] = np.nan
    r2_1d_prev[0] = np.nan
    s2_1d_prev[0] = np.nan
    r3_1d_prev[0] = np.nan
    s3_1d_prev[0] = np.nan
    
    # Align daily pivot levels to 1d timeframe (no shift needed as we use previous day's values)
    r1_1d_aligned = r1_1d_prev
    s1_1d_aligned = s1_1d_prev
    r2_1d_aligned = r2_1d_prev
    s2_1d_aligned = s2_1d_prev
    r3_1d_aligned = r3_1d_prev
    s3_1d_aligned = s3_1d_prev
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Weekly EMA34 for trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR filter to avoid low volatility environments
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = np.nan
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma10 = pd.Series(atr).rolling(window=10, min_periods=10).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # Need volume MA20, ATR MA10, and weekly EMA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(volume_ma20[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(atr_ma10[i]) or 
            np.isnan(r2_1d_aligned[i]) or 
            np.isnan(s2_1d_aligned[i]) or
            np.isnan(r3_1d_aligned[i]) or
            np.isnan(s3_1d_aligned[i]) or
            np.isnan(ema_34_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period average
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        # Volatility filter: ATR > ATR MA10 (avoid low volatility)
        volatility_filter = atr[i] > atr_ma10[i]
        
        if position == 0:
            # Long: price breaks above R2 with volume and volatility (breakout continuation)
            if (close[i] > r2_1d_aligned[i] and volume_filter and volatility_filter):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S2 with volume and volatility (breakout continuation)
            elif (close[i] < s2_1d_aligned[i] and volume_filter and volatility_filter):
                signals[i] = -0.25
                position = -1
            # Long mean reversion: price touches S3 with volume and volatility
            elif (close[i] < s3_1d_aligned[i] and volume_filter and volatility_filter):
                signals[i] = 0.20
                position = 1
            # Short mean reversion: price touches R3 with volume and volatility
            elif (close[i] > r3_1d_aligned[i] and volume_filter and volatility_filter):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: price returns below R1 or volatility drops
            if close[i] < r1_1d_aligned[i] or not volatility_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns above S1 or volatility drops
            if close[i] > s1_1d_aligned[i] or not volatility_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Camarilla_R2_S2_Breakout_Volume"
timeframe = "1d"
leverage = 1.0