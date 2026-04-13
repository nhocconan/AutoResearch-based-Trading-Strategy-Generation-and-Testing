#!/usr/bin/env python3
"""
4h_1D_Camarilla_Pivot_Breakout_Volume_Confirmation_v10
Hypothesis: Camarilla pivot levels from daily timeframe provide strong support/resistance.
Breakout above/below key levels (H3/L3) with volume confirmation (1d volume > 1.5x 20-day avg) 
and trend filter (ADX > 25 on 1d) captures institutional interest. 
Exit on reversal to pivot point (H4/L4) or trend weakness (ADX < 20).
Designed for 4h timeframe to target 20-35 trades/year, effective in both trending and ranging markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get daily data for Camarilla pivots and volume/ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous day
    # Formula: Based on previous day's high, low, close
    phigh = df_1d['high'].values
    plow = df_1d['low'].values
    pclose = df_1d['close'].values
    
    # Calculate pivot and levels
    pivot = (phigh + plow + pclose) / 3
    range_val = phigh - plow
    
    # Camarilla levels
    H4 = pclose + range_val * 1.1 / 2
    H3 = pclose + range_val * 1.1 / 4
    H2 = pclose + range_val * 1.1 / 6
    H1 = pclose + range_val * 1.1 / 12
    L1 = pclose - range_val * 1.1 / 12
    L2 = pclose - range_val * 1.1 / 6
    L3 = pclose - range_val * 1.1 / 4
    L4 = pclose - range_val * 1.1 / 2
    
    # Align daily levels to 4h timeframe
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    
    # Daily volume and average
    vol_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean()
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20.values)
    
    # Daily ADX calculation (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Wilder's smoothing function
    def wilder_smooth(arr, period):
        result = np.full_like(arr, np.nan, dtype=float)
        if len(arr) < period:
            return result
        result[period-1] = np.nansum(arr[:period])
        for i in range(period, len(arr)):
            result[i] = result[i-1] - (result[i-1]/period) + arr[i]
        return result
    
    tr_14 = wilder_smooth(tr, 14)
    dm_plus_14 = wilder_smooth(dm_plus, 14)
    dm_minus_14 = wilder_smooth(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(tr_14 != 0, 100 * dm_plus_14 / tr_14, 0)
    di_minus = np.where(tr_14 != 0, 100 * dm_minus_14 / tr_14, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = wilder_smooth(dx, 14)
    
    # Align daily volume and ADX to 4h
    vol_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_1d)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if any required data is not ready
        if (np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or 
            np.isnan(vol_1d_aligned[i]) or np.isnan(vol_ma_20_aligned[i]) or 
            np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume condition: current 1d volume > 1.5x 20-day average
        vol_condition = vol_1d_aligned[i] > (vol_ma_20_aligned[i] * 1.5)
        
        # ADX condition: trending market
        adx_condition = adx_aligned[i] > 25
        
        # Breakout conditions
        long_breakout = close[i] > H3_aligned[i]
        short_breakout = close[i] < L3_aligned[i]
        
        # Exit conditions
        long_exit = close[i] < L4_aligned[i]  # Exit at L4 (strong support)
        short_exit = close[i] > H4_aligned[i]  # Exit at H4 (strong resistance)
        trend_weak = adx_aligned[i] < 20
        
        if position == 0:
            if long_breakout and vol_condition and adx_condition:
                position = 1
                signals[i] = position_size
            elif short_breakout and vol_condition and adx_condition:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            if long_exit or trend_weak:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            if short_exit or trend_weak:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1D_Camarilla_Pivot_Breakout_Volume_Confirmation_v10"
timeframe = "4h"
leverage = 1.0