#!/usr/bin/env python3
"""
1d_Camarilla_R1_S1_Breakout_1wTrend_VolumeSpike_v2
Hypothesis: Uses 1w Camarilla pivot levels (R1/S1) for breakout entries on 1d timeframe with 1w EMA34 trend filter and volume confirmation.
Uses tighter volume threshold (3.0x avg) and requires close > open for confirmation to reduce false breakouts.
Long when price closes above R1 AND 1w close > EMA34 (uptrend) AND volume > 3.0 * 20-period average AND close > open.
Short when price closes below S1 AND 1w close < EMA34 (downtrend) AND volume > 3.0 * 20-period average AND close < open.
Exit when price returns to the pivot level (R1 for longs, S1 for shorts) OR trend reverses.
Designed for 1d timeframe to achieve 30-100 total trades over 4 years with low fee drag.
Works in both bull and bear markets by following 1w trend while using Camarilla levels for precise breakout entries.
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
    open_ = prices['open'].values
    
    # Get 1w data for trend filter and Camarilla pivots
    df_1w = get_htf_data(prices, '1w')
    
    # 1w EMA34 for trend filter
    close_1w_series = pd.Series(df_1w['close'].values)
    ema_34_1w = close_1w_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate 1w Camarilla pivot levels: R1, S1
    # Camarilla formulas: R1 = close + 1.1*(high-low)*1.1/12, S1 = close - 1.1*(high-low)*1.1/12
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    camarilla_r1 = close_1w + 1.1 * (high_1w - low_1w) * 1.1 / 12
    camarilla_s1 = close_1w - 1.1 * (high_1w - low_1w) * 1.1 / 12
    
    # Align Camarilla levels to 1d timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s1)
    
    # Volume confirmation: current volume > 3.0 * 20-period average (tighter)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (3.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need 1w EMA34 (34), volume avg (20)
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_aligned[i]) or np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        open_val = open_[i]
        high_val = high[i]
        low_val = low[i]
        ema_val = ema_34_aligned[i]
        r1_level = camarilla_r1_aligned[i]
        s1_level = camarilla_s1_aligned[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Look for entry: breakout of Camarilla R1/S1 with 1w trend filter AND volume AND directional candle
            # Long: price closes above R1 (minor resistance) AND 1w uptrend AND volume AND bullish candle
            long_condition = (close_val > r1_level) and (open_val > r1_level) and (close_val > ema_val) and vol_conf and (close_val > open_val)
            # Short: price closes below S1 (minor support) AND 1w downtrend AND volume AND bearish candle
            short_condition = (close_val < s1_level) and (open_val < s1_level) and (close_val < ema_val) and vol_conf and (close_val < open_val)
            
            if long_condition:
                signals[i] = size
                position = 1
            elif short_condition:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long when price returns to R1 level OR trend breaks
            exit_condition = (close_val <= r1_level) or (close_val < ema_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short when price returns to S1 level OR trend breaks
            exit_condition = (close_val >= s1_level) or (close_val > ema_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_Camarilla_R1_S1_Breakout_1wTrend_VolumeSpike_v2"
timeframe = "1d"
leverage = 1.0