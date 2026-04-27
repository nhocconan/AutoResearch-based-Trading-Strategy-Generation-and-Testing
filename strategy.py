#!/usr/bin/env python3
"""
12h_Camarilla_R3_S3_Breakout_1dTrend_VolumeConfirm
Hypothesis: Uses 12h timeframe with Camarilla R3/S3 levels from 1d data for key support/resistance.
Enter long when price breaks above Camarilla R3 AND 1d close > EMA34 (uptrend) AND volume > 1.5 * 20-period average.
Enter short when price breaks below Camarilla S3 AND 1d close < EMA34 (downtrend) AND volume > 1.5 * 20-period average.
Exit when price returns to the Camarilla pivot level (PP) OR trend reverses.
Camarilla pivots identify significant intraday support/resistance levels that work well in ranging and trending markets.
Combined with 1d trend filter and volume confirmation, this should work in both bull and bear markets by following higher timeframe structure.
Target: 50-150 total trades over 4 years (12-37/year) with 0.25 position size to manage drawdown.
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
    
    # Get 1d data for Camarilla calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA34 for trend filter
    close_1d_series = pd.Series(df_1d['close'].values)
    ema_34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from 1d OHLC
    # Camarilla equations: PP = (H+L+C)/3, Range = H-L
    # R3 = PP + (H-L) * 1.1/2, S3 = PP - (H-L) * 1.1/2
    typical_price_1d = (df_1d['high'].values + df_1d['low'].values + df_1d['close'].values) / 3.0
    range_1d = df_1d['high'].values - df_1d['low'].values
    camarilla_pp = typical_price_1d
    camarilla_r3 = camarilla_pp + (range_1d * 1.1 / 2.0)
    camarilla_s3 = camarilla_pp - (range_1d * 1.1 / 2.0)
    
    # Align Camarilla levels to 12h timeframe (no extra delay needed for pivot levels)
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pp)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need 1d EMA34 (34), volume avg (20), Camarilla calculation (need 1d OHLC)
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_aligned[i]) or np.isnan(camarilla_pp_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        ema_val = ema_34_aligned[i]
        pp_level = camarilla_pp_aligned[i]
        r3_level = camarilla_r3_aligned[i]
        s3_level = camarilla_s3_aligned[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Look for entry: breakout of Camarilla levels with 1d trend filter AND volume
            # Long: price breaks above Camarilla R3 (resistance) AND 1d uptrend AND volume
            long_condition = (close_val > r3_level) and (close_val > ema_val) and vol_conf
            # Short: price breaks below Camarilla S3 (support) AND 1d downtrend AND volume
            short_condition = (close_val < s3_level) and (close_val < ema_val) and vol_conf
            
            if long_condition:
                signals[i] = size
                position = 1
            elif short_condition:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long when price returns to Camarilla pivot level OR trend breaks
            exit_condition = (close_val <= pp_level) or (close_val < ema_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short when price returns to Camarilla pivot level OR trend breaks
            exit_condition = (close_val >= pp_level) or (close_val > ema_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Camarilla_R3_S3_Breakout_1dTrend_VolumeConfirm"
timeframe = "12h"
leverage = 1.0