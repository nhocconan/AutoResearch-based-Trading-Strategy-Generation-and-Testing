#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R1/S1 breakout with 1d ADX filter and volume confirmation.
# Camarilla levels provide precise support/resistance derived from prior day's range.
# Breakout above R1 (bullish) or below S1 (bearish) with volume > 2x 20-period average confirms conviction.
# 1d ADX > 25 ensures we trade only in strong trending markets to avoid whipsaws in ranging conditions.
# Discrete position sizing (0.25) minimizes trade frequency and fee drag.
# Target: 12-37 trades/year (50-150 total over 4 years) to balance opportunity and cost.
name = "12h_Camarilla_R1S1_Breakout_1dADX_Volume"
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
    
    # Get 1d data for Camarilla pivot calculation (needs prior day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels for each 1d bar: based on prior day's H, L, C
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Prior day's values (shift by 1 to avoid look-ahead)
    phigh = np.roll(high_1d, 1)
    plow = np.roll(low_1d, 1)
    pclose = np.roll(close_1d, 1)
    # First bar has no prior day
    phigh[0] = np.nan
    plow[0] = np.nan
    pclose[0] = np.nan
    
    # Calculate Camarilla R1 and S1 for prior day
    camarilla_r1 = pclose + (phigh - plow) * 1.1 / 12
    camarilla_s1 = pclose - (phigh - plow) * 1.1 / 12
    
    # Align Camarilla levels to 12h timeframe (using prior day's values, so already lagged)
    r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Get 1d data for ADX filter
    df_1d_adx = get_htf_data(prices, '1d')  # Reuse for ADX calculation
    
    # Calculate ADX on 1d data
    high_1d_adx = df_1d_adx['high'].values
    low_1d_adx = df_1d_adx['low'].values
    close_1d_adx = df_1d_adx['close'].values
    
    # True Range
    tr1 = high_1d_adx - low_1d_adx
    tr2 = np.abs(high_1d_adx - np.roll(close_1d_adx, 1))
    tr3 = np.abs(low_1d_adx - np.roll(close_1d_adx, 1))
    tr1[0] = np.nan  # First bar has no previous close
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Directional Movement
    up_move = np.diff(high_1d_adx, prepend=np.nan)
    down_move = -np.diff(low_1d_adx, prepend=np.nan)  # Negative of change
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    # First element remains nan due to prepend
    
    # Smoothed DM using Wilder's smoothing (alpha = 1/14)
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, ignore_nan=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, ignore_nan=False).mean().values
    
    # DI and DX
    plus_di = 100 * plus_dm_smooth / atr_1d
    minus_di = 100 * minus_dm_smooth / atr_1d
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx_1d = pd.Series(dx).ewm(alpha=1/14, adjust=False, ignore_nan=False).mean().values
    
    # Align ADX to 12h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d_adx, adx_1d)
    
    # Calculate volume spike: current volume > 2.0 * 20-period average volume
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(adx_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        r1 = r1_aligned[i]
        s1 = s1_aligned[i]
        
        # Breakout conditions
        bullish_breakout = price > r1
        bearish_breakout = price < s1
        
        # Strong trend filter: ADX > 25
        strong_trend = adx_1d_aligned[i] > 25
        
        if position == 0:
            # Long: Bullish breakout AND strong trend AND volume spike
            if bullish_breakout and strong_trend and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Bearish breakout AND strong trend AND volume spike
            elif bearish_breakout and strong_trend and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price breaks below S1 (reversal) OR ADX weakens
            if price < s1 or adx_1d_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price breaks above R1 (reversal) OR ADX weakens
            if price > r1 or adx_1d_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals