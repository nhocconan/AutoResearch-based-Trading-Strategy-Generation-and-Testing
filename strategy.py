#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d pivot points with volume confirmation and ADX trend filter
# 1-day pivot points (R1/S1 for breakouts, R2/S2 for reversals) provide key daily levels
# Breakout above R1 or below S1 with volume > 1.5x 20-period average indicates strong momentum
# ADX(14) > 25 ensures we only trade in trending markets, avoiding chop
# Works in bull/bear markets: breakouts capture trends, reversals capture pullbacks within trend
# Target: 75-200 total trades over 4 years (19-50/year) with 0.25 position sizing

name = "4h_1dPivot_R1S2_VolumeADX_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d pivot points ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous 1d bar's OHLC for pivot calculation
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Pivot point calculation
    # Pivot = (previous high + previous low + previous close) / 3
    pivot = (prev_high + prev_low + prev_close) / 3
    range_ = prev_high - prev_low
    
    # Support and Resistance levels
    r1 = pivot + (range_ * 1.0)
    r2 = pivot + (range_ * 2.0)
    s1 = pivot - (range_ * 1.0)
    s2 = pivot - (range_ * 2.0)
    
    # Align 1d levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    
    # Volume confirmation: >1.5x 20-period average (balanced to reduce trades)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    # ADX trend filter: >25 indicates trending market
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(np.maximum(tr1, tr2), tr3)])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    up_move = high[1:] - high[:-1]
    down_move = low[:-1] - low[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_filter = adx > 25
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(r1_aligned[i]) or np.isnan(r2_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(s2_aligned[i]) or np.isnan(volume_filter[i]) or np.isnan(adx_filter[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price breaks above R1 with volume confirmation and trend
            if close[i] > r1_aligned[i] and volume_filter[i] and adx_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short breakout: price breaks below S1 with volume confirmation and trend
            elif close[i] < s1_aligned[i] and volume_filter[i] and adx_filter[i]:
                signals[i] = -0.25
                position = -1
            # Long reversal: price rejects S2 with volume confirmation and trend
            elif close[i] < s2_aligned[i] and close[i] > s2_aligned[i] * 0.995 and volume_filter[i] and adx_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short reversal: price rejects R2 with volume confirmation and trend
            elif close[i] > r2_aligned[i] and close[i] < r2_aligned[i] * 1.005 and volume_filter[i] and adx_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below S1 (failed support) or reaches R2 (take profit)
            if close[i] < s1_aligned[i] or close[i] > r2_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above R1 (failed resistance) or reaches S2 (take profit)
            if close[i] > r1_aligned[i] or close[i] < s2_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals