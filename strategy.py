#!/usr/bin/env python3
"""
6h_Camarilla_R4_S4_Breakout_1dTrend_VolumeSpike_v1
Hypothesis: Breakouts at R4/S4 (extreme intraday levels) with 1d EMA34 trend alignment and volume spikes capture strong momentum moves. 
The 6h timeframe reduces noise vs lower timeframes while capturing intraday trends. 
Volume confirmation ensures breakouts have conviction, and trend filter avoids counter-trend trades. 
Target: 50-150 total trades over 4 years (12-37/year).
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
    
    # Get 1d data for Camarilla and trend
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Camarilla levels (R4, S4) from prior day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    range_1d = high_1d - low_1d
    camarilla_r4 = close_1d + 2.0 * range_1d  # R4
    camarilla_s4 = close_1d - 2.0 * range_1d  # S4
    
    # Calculate 1d EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Volume confirmation: current volume > 2.5 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.5 * vol_avg)
    
    # Align all indicators to primary timeframe (6h)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    volume_confirm_aligned = align_htf_to_ltf(prices, df_1d, volume_confirm)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    size = 0.25   # Position size: 25% of capital (discrete level)
    
    # Warmup: need Camarilla (1), EMA34 (34), volume avg (20)
    start_idx = max(1, 34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(volume_confirm_aligned[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        r4 = camarilla_r4_aligned[i]
        s4 = camarilla_s4_aligned[i]
        ema34 = ema34_1d_aligned[i]
        vol_conf = volume_confirm_aligned[i]
        
        if position == 0:
            # Determine trend alignment: price vs EMA34 (1d)
            uptrend = close_val > ema34
            downtrend = close_val < ema34
            
            if uptrend and vol_conf:
                # Long bias: long when price breaks above R4 with volume
                if close_val > r4:
                    signals[i] = size
                    position = 1
                    entry_price = close_val
            elif downtrend and vol_conf:
                # Short bias: short when price breaks below S4 with volume
                if close_val < s4:
                    signals[i] = -size
                    position = -1
                    entry_price = close_val
        elif position == 1:
            # Exit conditions: stoploss (3.0*ATR) or touch opposite level (S4)
            atr_approx = pd.Series(high - low).rolling(window=20, min_periods=20).mean().values[i]
            stop_loss = entry_price - 3.0 * atr_approx
            
            if close_val <= stop_loss:
                signals[i] = 0.0
                position = 0
            elif close_val < s4:  # Touch S4 (opposite level)
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit conditions: stoploss (3.0*ATR) or touch opposite level (R4)
            atr_approx = pd.Series(high - low).rolling(window=20, min_periods=20).mean().values[i]
            stop_loss = entry_price + 3.0 * atr_approx
            
            if close_val >= stop_loss:
                signals[i] = 0.0
                position = 0
            elif close_val > r4:  # Touch R4 (opposite level)
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Camarilla_R4_S4_Breakout_1dTrend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0