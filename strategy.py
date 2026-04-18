#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R1/S1 breakout with 1d volume confirmation and 1d ADX trend filter.
# Long when price breaks above R1 with volume > 1.5x 24-period average and ADX > 20.
# Short when price breaks below S1 with same conditions.
# Exit when price returns to pivot (PP) or reverses to opposite level.
# Uses 12h timeframe for lower frequency, 1d for volume/ADX filters to reduce noise.
# Designed for ~15-30 trades/year per symbol with strong edge in both bull and bear markets.
name = "12h_Camarilla_R1S1_Volume_ADX_Filter"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for volume and ADX filters
    df_1d = get_htf_data(prices, '1d')
    
    # Volume filter: current volume > 1.5 * 24-period average (24 * 1h = 1 day)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_filter = volume > (1.5 * vol_ma_24)
    
    # ADX(14) on 1d for trend strength filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # first period
    tr2[0] = high_1d[0] - close_1d[0]
    tr3[0] = low_1d[0] - close_1d[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    up_move[0] = 0
    down_move[0] = 0
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    plus_dm_14 = pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values
    minus_dm_14 = pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values
    
    # DI values
    plus_di_14 = 100 * plus_dm_14 / tr_14
    minus_di_14 = 100 * minus_dm_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14)
    adx_1d = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # 12h data for Camarilla pivot levels
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Camarilla pivot levels calculation
    # Pivot Point (PP) = (High + Low + Close) / 3
    pp_12h = (high_12h + low_12h + close_12h) / 3
    range_12h = high_12h - low_12h
    
    # Resistance and Support levels
    r1_12h = pp_12h + (range_12h * 1.0 / 12)
    s1_12h = pp_12h - (range_12h * 1.0 / 12)
    
    # Align Camarilla levels to 12h timeframe (already aligned as we're using 12h data directly)
    r1_12h_aligned = r1_12h  # Already at 12h frequency
    s1_12h_aligned = s1_12h  # Already at 12h frequency
    pp_12h_aligned = pp_12h  # Already at 12h frequency
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(vol_ma_24[i]) or
            np.isnan(r1_12h_aligned[i]) or np.isnan(s1_12h_aligned[i]) or
            np.isnan(pp_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        vol_filter = volume_filter[i]
        adx_val = adx_1d_aligned[i]
        r1_val = r1_12h_aligned[i]
        s1_val = s1_12h_aligned[i]
        pp_val = pp_12h_aligned[i]
        
        if position == 0:
            # Long: price breaks above R1 with volume and trend strength
            if close_val > r1_val and vol_filter and adx_val > 20:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume and trend strength
            elif close_val < s1_val and vol_filter and adx_val > 20:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to pivot or breaks below S1 (reversal)
            if close_val <= pp_val or close_val < s1_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to pivot or breaks above R1 (reversal)
            if close_val >= pp_val or close_val > r1_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals