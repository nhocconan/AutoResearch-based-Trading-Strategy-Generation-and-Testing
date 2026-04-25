#!/usr/bin/env python3
"""
1d_Camarilla_R3S3_Breakout_1wTrend_VolumeSpike
Hypothesis: Camarilla R3/S3 levels on 1d identify key reversal zones. A break above R3 with volume spike and 1w uptrend signals long; break below S3 with volume spike and 1w downtrend signals short. Uses 1w EMA34 for trend filter and discrete position sizing (0.25) to limit fee drag. Works in both bull and bear markets by capturing breakouts from statistically significant levels with trend and volume confirmation. Target: 7-25 trades/year per symbol.
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
    volume = prices['volume'].values
    
    # 1d data for Camarilla calculation (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels for 1d
    # Camarilla: R4 = close + ((high-low) * 1.1/2), R3 = close + ((high-low) * 1.1/4), etc.
    # We only need R3 and S3
    hl_range = df_1d['high'] - df_1d['low']
    close_1d = df_1d['close']
    camarilla_r3 = close_1d + (hl_range * 1.1 / 4)
    camarilla_s3 = close_1d - (hl_range * 1.1 / 4)
    
    # 1w EMA34 for trend filter (loaded ONCE)
    df_1w = get_htf_data(prices, '1w')
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Align Camarilla levels to 1d timeframe (no extra delay needed as they're based on completed 1d bar)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3.values)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3.values)
    
    # 1d volume spike: current volume > 2.0 * 20-period volume MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # track entry price for stoploss
    
    # Start index: need volume MA (20) + aligned HTF arrays
    start_idx = max(20, 0)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        if position == 0:
            # Long: price breaks above Camarilla R3 with volume spike and 1w uptrend
            long_breakout = (curr_close > camarilla_r3_aligned[i]) and vol_spike[i] and (curr_close > ema_34_1w_aligned[i])
            # Short: price breaks below Camarilla S3 with volume spike and 1w downtrend
            short_breakout = (curr_close < camarilla_s3_aligned[i]) and vol_spike[i] and (curr_close < ema_34_1w_aligned[i])
            
            if long_breakout:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_breakout:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price breaks below Camarilla S3 OR trend turns down
            if (curr_close < camarilla_s3_aligned[i]) or (curr_close < ema_34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price breaks above Camarilla R3 OR trend turns up
            if (curr_close > camarilla_r3_aligned[i]) or (curr_close > ema_34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Camarilla_R3S3_Breakout_1wTrend_VolumeSpike"
timeframe = "1d"
leverage = 1.0