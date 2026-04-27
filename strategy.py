#!/usr/bin/env python3
"""
6h_Camarilla_R4_S4_Breakout_1wTrend_VolumeConfirm
Hypothesis: Uses weekly Camarilla pivot levels (R4/S4) for breakout entries on 6h timeframe.
Enter long when price breaks above weekly R4 AND 1w close > EMA34 (uptrend) AND volume > 2.0 * 20-period average.
Enter short when price breaks below weekly S4 AND 1w close < EMA34 (downtrend) AND volume > 2.0 * 20-period average.
Exit when price returns to weekly pivot (PP) level OR trend reverses.
Camarilla R4/S4 represent extreme breakout levels; weekly trend filter ensures alignment with higher timeframe structure.
High volume threshold (2.0x) filters weak breakouts. Target: 50-150 total trades over 4 years (12-37/year) with 0.25 position size.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d and 1w data for Camarilla pivots and weekly trend filter
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # 1w EMA34 for trend filter
    close_1w_series = pd.Series(df_1w['close'].values)
    ema_34_1w = close_1w_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate Camarilla pivots on 1d data (using previous day's OHLC)
    # Camarilla levels: R4 = C + ((H-L) * 1.1/2), S4 = C - ((H-L) * 1.1/2), PP = (H+L+C)/3
    # We need previous day's data to calculate today's levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Shift by 1 to get previous day's OHLC for today's Camarilla levels
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    # First value will be invalid (rolled from last), set to nan
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Calculate Camarilla levels
    camarilla_pp = (prev_high + prev_low + prev_close) / 3.0
    camarilla_range = prev_high - prev_low
    camarilla_r4 = camarilla_pp + (camarilla_range * 1.1 / 2.0)
    camarilla_s4 = camarilla_pp - (camarilla_range * 1.1 / 2.0)
    
    # Align 1d Camarilla levels to 6h timeframe
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pp)
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need 1w EMA34 (34), volume avg (20), 1d data shifted (1)
    start_idx = max(34, 20, 1)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(camarilla_r4_aligned[i]) or 
            np.isnan(camarilla_s4_aligned[i]) or np.isnan(camarilla_pp_aligned[i]) or 
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema_val = ema_34_1w_aligned[i]
        r4_level = camarilla_r4_aligned[i]
        s4_level = camarilla_s4_aligned[i]
        pp_level = camarilla_pp_aligned[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Look for entry: breakout of Camarilla R4/S4 levels with 1w trend filter AND volume
            # Long: price breaks above R4 AND 1w uptrend AND volume
            long_condition = (close_val > r4_level) and (close_val > ema_val) and vol_conf
            # Short: price breaks below S4 AND 1w downtrend AND volume
            short_condition = (close_val < s4_level) and (close_val < ema_val) and vol_conf
            
            if long_condition:
                signals[i] = size
                position = 1
            elif short_condition:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long when price returns to pivot level OR trend breaks
            exit_condition = (close_val <= pp_level) or (close_val < ema_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short when price returns to pivot level OR trend breaks
            exit_condition = (close_val >= pp_level) or (close_val > ema_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Camarilla_R4_S4_Breakout_1wTrend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0