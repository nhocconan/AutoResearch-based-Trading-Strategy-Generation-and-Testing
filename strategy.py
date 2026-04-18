#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla Pivot R1/S1 breakout with volume confirmation and 1d ADX trend filter.
# Long when price breaks above R1 with volume > 1.5x 24-period average and ADX > 25.
# Short when price breaks below S1 with same conditions.
# Exit when price returns to the 1d pivot point (PP).
# Uses 1d pivot levels for structure, volume for conviction, and ADX to avoid ranging markets.
# Designed for ~15-30 trades/year on 12h timeframe.
name = "12h_Pivot_R1_S1_Breakout_Volume_ADX"
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
    
    # 1d data for pivot points and ADX
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily pivot points
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pp = (high_1d + low_1d + close_1d) / 3.0
    r1 = 2 * pp - low_1d
    s1 = 2 * pp - high_1d
    r2 = pp + (high_1d - low_1d)
    s2 = pp - (high_1d - low_1d)
    
    # Align pivot levels to 12h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # ADX(14) on 1d for trend strength
    # Calculate True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - close_1d)
    tr3 = np.abs(low_1d - close_1d)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Directional Movement
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smooth with Wilder's smoothing (equivalent to EMA with alpha=1/14)
    def wilders_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            result[period-1] = np.nansum(data[:period])
            for i in range(period, len(data)):
                result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    tr_14 = wilders_smooth(tr, 14)
    plus_dm_14 = wilders_smooth(plus_dm, 14)
    minus_dm_14 = wilders_smooth(minus_dm, 14)
    
    # Avoid division by zero
    plus_di_14 = np.where(tr_14 != 0, (plus_dm_14 / tr_14) * 100, 0)
    minus_di_14 = np.where(tr_14 != 0, (minus_dm_14 / tr_14) * 100, 0)
    
    dx = np.where((plus_di_14 + minus_di_14) != 0, 
                  np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14) * 100, 0)
    adx_1d = wilders_smooth(dx, 14)
    
    # Align ADX to 12h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Volume filter: current volume > 1.5x 24-period average (24 * 12h = 12 days)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_filter = volume > (1.5 * vol_ma_24)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(adx_1d_aligned[i]) or np.isnan(vol_ma_24[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        pp_val = pp_aligned[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        adx_val = adx_1d_aligned[i]
        vol_filter = volume_filter[i]
        
        if position == 0:
            # Long: price breaks above R1 with volume and trend
            if close_val > r1_val and vol_filter and adx_val > 25:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume and trend
            elif close_val < s1_val and vol_filter and adx_val > 25:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to pivot point
            if close_val <= pp_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to pivot point
            if close_val >= pp_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals