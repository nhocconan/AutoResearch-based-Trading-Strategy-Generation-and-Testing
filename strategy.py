#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_4hTrend_VolumeSpike
Hypothesis: Uses 4h Camarilla pivot levels (R1/S1) for breakout entries on 4h timeframe.
Enter long when price breaks above 4h R1 AND 4h close > 4h EMA20 (uptrend) AND volume > 1.8 * 20-period average.
Enter short when price breaks below 4h S1 AND 4h close < 4h EMA20 (downtrend) AND volume > 1.8 * 20-period average.
Exit when price returns to 4h pivot (PP) level OR trend reverses.
Camarilla R1/S1 represent strong breakout levels; 4h trend filter ensures alignment with higher timeframe structure (12h).
High volume threshold (1.8x) filters weak breakouts. Target: 75-150 total trades over 4 years (19-38/year) with 0.28 position size.
Designed to work in both bull and bear markets via 12h trend filter and breakout logic.
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
    
    # Get 4h data for Camarilla pivots and 4h trend filter
    df_4h = get_htf_data(prices, '4h')
    
    # 4h EMA20 for trend filter
    close_4h_series = pd.Series(df_4h['close'].values)
    ema_20_4h = close_4h_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # Calculate Camarilla pivots on 4h data (using previous 4h bar's OHLC)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Shift by 1 to get previous 4h bar's OHLC for current Camarilla levels
    prev_high = np.roll(high_4h, 1)
    prev_low = np.roll(low_4h, 1)
    prev_close = np.roll(close_4h, 1)
    # First value will be invalid (rolled from last), set to nan
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Calculate Camarilla levels
    camarilla_pp = (prev_high + prev_low + prev_close) / 3.0
    camarilla_range = prev_high - prev_low
    camarilla_r1 = camarilla_pp + (camarilla_range * 1.1 / 4.0)
    camarilla_s1 = camarilla_pp - (camarilla_range * 1.1 / 4.0)
    
    # Align 4h Camarilla levels to 4h timeframe (identity alignment)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s1)
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_4h, camarilla_pp)
    
    # Get 12h data for higher timeframe trend filter
    df_12h = get_htf_data(prices, '12h')
    
    # 12h EMA50 for trend filter
    close_12h_series = pd.Series(df_12h['close'].values)
    ema_50_12h = close_12h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume confirmation: current volume > 1.8 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.8 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.28   # Position size: 28% of capital
    
    # Warmup: need 4h EMA20 (20), 12h EMA50 (50), volume avg (20), 4h data shifted (1)
    start_idx = max(50, 20, 20, 1)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_20_4h_aligned[i]) or np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(camarilla_pp_aligned[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema_4h_val = ema_20_4h_aligned[i]
        ema_12h_val = ema_50_12h_aligned[i]
        r1_level = camarilla_r1_aligned[i]
        s1_level = camarilla_s1_aligned[i]
        pp_level = camarilla_pp_aligned[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Look for entry: breakout of Camarilla R1/S1 levels with 4h AND 12h trend filter AND volume
            # Long: price breaks above R1 AND 4h uptrend AND 12h uptrend AND volume
            long_condition = (close_val > r1_level) and (close_val > ema_4h_val) and (close_val > ema_12h_val) and vol_conf
            # Short: price breaks below S1 AND 4h downtrend AND 12h downtrend AND volume
            short_condition = (close_val < s1_level) and (close_val < ema_4h_val) and (close_val < ema_12h_val) and vol_conf
            
            if long_condition:
                signals[i] = size
                position = 1
            elif short_condition:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long when price returns to pivot level OR 4h trend breaks OR 12h trend breaks
            exit_condition = (close_val <= pp_level) or (close_val < ema_4h_val) or (close_val < ema_12h_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short when price returns to pivot level OR 4h trend breaks OR 12h trend breaks
            exit_condition = (close_val >= pp_level) or (close_val > ema_4h_val) or (close_val > ema_12h_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_4hTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0