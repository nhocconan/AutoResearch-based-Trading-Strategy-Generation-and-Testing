#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike
Hypothesis: Uses 4h Camarilla R3/S3 breakouts filtered by 1d EMA34 trend and volume spike (>2x average).
Enter long when price breaks above 4h R3 AND 1d close > 1d EMA34 (uptrend) AND volume > 2x average.
Enter short when price breaks below 4h S3 AND 1d close < 1d EMA34 (downtrend) AND volume > 2x average.
Exit when price re-enters the Camarilla H3-L3 range (mean-reversion zone) OR 1d trend breaks.
Designed for 4h timeframe with tight entries to avoid fee drag: target 19-50 trades/year.
Works in both bull and bear markets via 1d trend filter and volume confirmation to avoid false signals.
Camarilla levels provide institutional support/resistance with high probability reaction zones.
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
    
    # Get 4h data for Camarilla calculation
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate Camarilla levels for 4h timeframe
    # Based on previous 4h bar's high, low, close
    ph = df_4h['high'].values  # previous high
    pl = df_4h['low'].values   # previous low
    pc = df_4h['close'].values # previous close
    
    # Camarilla formulas
    range_hl = ph - pl
    r3 = pc + (range_hl * 1.1 / 4)
    r2 = pc + (range_hl * 1.1 / 6)
    r1 = pc + (range_hl * 1.1 / 12)
    s1 = pc - (range_hl * 1.1 / 12)
    s2 = pc - (range_hl * 1.1 / 6)
    s3 = pc - (range_hl * 1.1 / 4)
    h3 = pc + (range_hl * 1.1 / 4)  # Same as R3
    h2 = pc + (range_hl * 1.1 / 6)  # Same as R2
    h1 = pc + (range_hl * 1.1 / 12) # Same as R1
    l3 = pc - (range_hl * 1.1 / 4)  # Same as S3
    l2 = pc - (range_hl * 1.1 / 6)  # Same as S2
    l1 = pc - (range_hl * 1.1 / 12) # Same as S1
    
    # Align 4h indicators to 4h timeframe (identity alignment)
    r3_aligned = align_htf_to_ltf(prices, df_4h, r3)
    s3_aligned = align_htf_to_ltf(prices, df_4h, s3)
    h3_aligned = align_htf_to_ltf(prices, df_4h, h3)
    l3_aligned = align_htf_to_ltf(prices, df_4h, l3)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA34 for trend filter
    close_1d_series = pd.Series(df_1d['close'].values)
    ema_34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: current volume > 2 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need 4h data (1 bar) + 1d EMA34 (34) + volume avg (20)
    start_idx = max(1, 34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or 
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema_1d_val = ema_34_1d_aligned[i]
        r3_val = r3_aligned[i]
        s3_val = s3_aligned[i]
        h3_val = h3_aligned[i]
        l3_val = l3_aligned[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Look for entry: Camarilla breakout with trend and volume
            # Long: Close > R3 AND 1d uptrend AND volume
            long_condition = (close_val > r3_val) and (close_val > ema_1d_val) and vol_conf
            # Short: Close < S3 AND 1d downtrend AND volume
            short_condition = (close_val < s3_val) and (close_val < ema_1d_val) and vol_conf
            
            if long_condition:
                signals[i] = size
                position = 1
            elif short_condition:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long when price < H3 (mean reversion) OR 1d trend breaks
            exit_condition = (close_val < h3_val) or (close_val < ema_1d_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short when price > L3 (mean reversion) OR 1d trend breaks
            exit_condition = (close_val > l3_val) or (close_val > ema_1d_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0