#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R1/S1 breakout with 1d volume confirmation and ADX trend filter
# Camarilla levels provide precise intraday support/resistance from prior day
# Breakout of R1 (resistance 1) or S1 (support 1) with volume confirms momentum
# 1d ADX > 25 ensures we only trade in trending markets, avoiding whipsaws in ranges
# Target: 50-150 total trades over 4 years (12-37/year) with disciplined entries
name = "12h_Camarilla_R1S1_Breakout_Volume_ADX"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla calculation and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough data for ADX
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    # R4 = C + ((H-L) * 1.5000)
    # R3 = C + ((H-L) * 1.2500)
    # R2 = C + ((H-L) * 1.1666)
    # R1 = C + ((H-L) * 1.0833)
    # PP = (H + L + C) / 3
    # S1 = C - ((H-L) * 1.0833)
    # S2 = C - ((H-L) * 1.1666)
    # S3 = C - ((H-L) * 1.2500)
    # S4 = C - ((H-L) * 1.5000)
    
    # We need previous day's H, L, C to calculate today's levels
    # So we shift the 1d data by 1 to get previous day's values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Calculate Camarilla levels for current day based on previous day
    camarilla_r1 = prev_close + ((prev_high - prev_low) * 1.0833)
    camarilla_s1 = prev_close - ((prev_high - prev_low) * 1.0833)
    
    # Align to 12h timeframe (these levels are constant throughout the 12h period)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # 1d ADX for trend filter
    # Calculate directional movement
    high_diff = df_1d['high'].diff()
    low_diff = df_1d['low'].diff().multiply(-1)
    
    plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
    minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
    
    # True Range
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = np.abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Smooth with Wilder's smoothing (equivalent to EMA with alpha=1/period)
    period = 14
    atr = np.zeros(len(df_1d))
    atr[period-1] = tr.iloc[:period].mean()
    for i in range(period, len(df_1d)):
        atr[i] = (atr[i-1] * (period-1) + tr.iloc[i]) / period
    
    plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=1/period, adjust=False).mean() / atr)
    minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=1/period, adjust=False).mean() / atr)
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=1/period, adjust=False).mean().values
    
    # Align ADX to 12h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume confirmation: volume > 1.5 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data for all indicators
    start_idx = max(30, 20)  # 30 for ADX stabilization, 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Only trade when ADX > 25 (trending market)
        if adx_aligned[i] <= 25:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R1 with volume confirmation
            if close[i] > camarilla_r1_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume confirmation
            elif close[i] < camarilla_s1_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below S1 (reversal signal)
            if close[i] < camarilla_s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price breaks above R1 (reversal signal)
            if close[i] > camarilla_r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals