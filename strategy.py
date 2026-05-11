#!/usr/bin/env python3
name = "1d_WEEKLY_CAMARILLA_BREAKOUT_1wEMA34_VOLUME_CONFIRM"
timeframe = "1d"
leverage = 1.0

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
    
    # Weekly EMA34 for trend filter (HTF) - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    ema34_1w = pd.Series(df_1w['close']).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Weekly high/low/close for Camarilla levels
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Camarilla levels: R3, S3
    hl_range = high_1w - low_1w
    r3 = close_1w + hl_range * 1.25
    s3 = close_1w - hl_range * 1.25
    
    # Align Camarilla levels to daily timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    
    # Volume filter: 20-period EMA for confirmation
    vol_ema20 = pd.Series(volume).ewm(span=20, min_periods=20, adjust=False).mean().values
    volume_ok = volume > vol_ema20 * 2.0
    
    # Fixed position size
    position_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(ema34_1w_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(volume_ok[i])):
            if position == 1:
                signals[i] = 0.0
            elif position == -1:
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        # Conditions
        price_above_ema1w = close[i] > ema34_1w_aligned[i]
        price_below_ema1w = close[i] < ema34_1w_aligned[i]
        breakout_long = close[i] > r3_aligned[i]
        breakout_short = close[i] < s3_aligned[i]
        
        if position == 0:
            # Long: Price breaks above R3 + above weekly EMA34 + volume confirmation
            if breakout_long and price_above_ema1w and volume_ok[i]:
                signals[i] = position_size
                position = 1
            # Short: Price breaks below S3 + below weekly EMA34 + volume confirmation
            elif breakout_short and price_below_ema1w and volume_ok[i]:
                signals[i] = -position_size
                position = -1
        else:
            # Exit conditions - reverse of entry
            if position == 1:
                # Exit: Price breaks below S3 OR trend reverses (below weekly EMA34)
                if close[i] < s3_aligned[i] or close[i] < ema34_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = position_size
            elif position == -1:
                # Exit: Price breaks above R3 OR trend reverses (above weekly EMA34)
                if close[i] > r3_aligned[i] or close[i] > ema34_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -position_size
    
    return signals