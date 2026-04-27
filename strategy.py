#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeSpike
Hypothesis: Uses 4h Camarilla pivot levels (R1/S1) for breakout entries on 1h timeframe.
Enter long when price breaks above 4h R1 AND 4h close > EMA20 (uptrend) AND volume > 1.5 * 20-period average.
Enter short when price breaks below 4h S1 AND 4h close < EMA20 (downtrend) AND volume > 1.5 * 20-period average.
Exit when price returns to 4h pivot (PP) level OR trend reverses.
Session filter: only trade 08-20 UTC to avoid low-liquidity hours.
Position size: 0.20 (20% of capital).
Designed for 1h timeframe with 4h trend filter to reduce overtrading and improve Sharpe.
Target: 60-150 total trades over 4 years (15-37/year) with discrete position sizing to minimize fee drag.
Works in both bull and bear markets via 4h trend alignment and volume confirmation.
"""

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
    
    # Session filter: 08-20 UTC (precomputed for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for Camarilla pivots and trend filter
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
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Calculate Camarilla levels (R1/S1 = PP ± (H-L)*1.1/2, PP = (H+L+C)/3)
    camarilla_pp = (prev_high + prev_low + prev_close) / 3.0
    camarilla_range = prev_high - prev_low
    camarilla_r1 = camarilla_pp + (camarilla_range * 1.1 / 2.0)
    camarilla_s1 = camarilla_pp - (camarilla_range * 1.1 / 2.0)
    
    # Align 4h Camarilla levels to 1h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s1)
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_4h, camarilla_pp)
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.20   # Position size: 20% of capital
    
    # Warmup: need 4h EMA20 (20), volume avg (20), 4h data shifted (1)
    start_idx = max(20, 20, 1)
    
    for i in range(start_idx, n):
        # Skip if outside trading session or data not ready
        if not in_session[i] or \
           (np.isnan(ema_20_4h_aligned[i]) or np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or np.isnan(camarilla_pp_aligned[i]) or 
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema_val = ema_20_4h_aligned[i]
        r1_level = camarilla_r1_aligned[i]
        s1_level = camarilla_s1_aligned[i]
        pp_level = camarilla_pp_aligned[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Look for entry: breakout of Camarilla R1/S1 levels with 4h trend filter AND volume
            # Long: price breaks above R1 AND 4h uptrend AND volume
            long_condition = (close_val > r1_level) and (close_val > ema_val) and vol_conf
            # Short: price breaks below S1 AND 4h downtrend AND volume
            short_condition = (close_val < s1_level) and (close_val < ema_val) and vol_conf
            
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

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeSpike"
timeframe = "1h"
leverage = 1.0