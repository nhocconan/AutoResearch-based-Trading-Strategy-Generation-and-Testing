#!/usr/bin/env python3
"""
6h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike
Hypothesis: Camarilla R3/S3 breakouts on 6h timeframe with 1d trend filter (price >/ < EMA34) and volume spike confirmation capture institutional participation in trending markets. This strategy targets breakouts from key daily pivot levels with momentum confirmation, working in both bull (breakouts above R3 in uptrend) and bear (breakdowns below S3 in downtrend) regimes. Discrete sizing (0.25) and strict volume confirmation limit overtrading. Target: 50-150 total trades over 4 years.
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
    
    # Get 1d data for Camarilla pivot calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Camarilla levels from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's Camarilla levels
    prev_high = pd.Series(high_1d).shift(1).values
    prev_low = pd.Series(low_1d).shift(1).values
    prev_close = pd.Series(close_1d).shift(1).values
    
    # Camarilla calculations
    range_1d = prev_high - prev_low
    camarilla_h5 = prev_close + (range_1d * 1.1 / 2)  # H5
    camarilla_h4 = prev_close + (range_1d * 1.1 / 4)  # H4
    camarilla_h3 = prev_close + (range_1d * 1.1 / 6)  # H3 (R3)
    camarilla_l3 = prev_close - (range_1d * 1.1 / 6)  # L3 (S3)
    camarilla_l4 = prev_close - (range_1d * 1.1 / 4)  # L4
    camarilla_l5 = prev_close - (range_1d * 1.1 / 2)  # L5
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align all indicators to primary timeframe (6h)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)   # R3
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)   # S3
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: current volume > 2.0 * 20-period average (6h equivalent)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    size = 0.25   # Position size: 25% of capital (discrete level)
    
    # Warmup: need Camarilla (1d), EMA34 (34), volume avg (20)
    start_idx = max(35, 20)  # +1 for shift
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        r3_level = camarilla_h3_aligned[i]
        s3_level = camarilla_l3_aligned[i]
        ema_trend = ema_34_aligned[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Long: price breaks above R3 with volume confirmation in uptrend (price > EMA34)
            if (close_val > r3_level) and vol_conf and (close_val > ema_trend):
                signals[i] = size
                position = 1
                entry_price = close_val
            # Short: price breaks below S3 with volume confirmation in downtrend (price < EMA34)
            elif (close_val < s3_level) and vol_conf and (close_val < ema_trend):
                signals[i] = -size
                position = -1
                entry_price = close_val
        elif position == 1:
            # Exit: stoploss (2.5*ATR) or price re-enters Camarilla H3-L3 range
            atr_approx = pd.Series(high - low).rolling(window=14, min_periods=14).mean().values[i]
            stop_loss = entry_price - 2.5 * atr_approx
            
            if close_val <= stop_loss:
                signals[i] = 0.0
                position = 0
            elif close_val < r3_level and close_val > s3_level:  # Re-entered H3-L3 range
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit: stoploss (2.5*ATR) or price re-enters Camarilla H3-L3 range
            atr_approx = pd.Series(high - low).rolling(window=14, min_periods=14).mean().values[i]
            stop_loss = entry_price + 2.5 * atr_approx
            
            if close_val >= stop_loss:
                signals[i] = 0.0
                position = 0
            elif close_val < r3_level and close_val > s3_level:  # Re-entered H3-L3 range
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0