#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike.
Long when price breaks above Camarilla R3 AND 1d EMA34 rising AND volume > 2.0x 20-period average.
Short when price breaks below Camarilla S3 AND 1d EMA34 falling AND volume > 2.0x 20-period average.
Exit when price touches opposite Camarilla level (R2/S2) or EMA34 direction reverses.
Uses 1d HTF for EMA34 trend to filter whipsaws. Target: 75-150 total trades over 4 years.
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
    
    # Calculate 1d EMA34 for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 4h Camarilla levels (based on previous day's OHLC)
    # We need daily OHLC from 1d data aligned to 4h bars
    # Camarilla: R3 = C + (H-L)*1.1/2, S3 = C - (H-L)*1.1/2
    # where C, H, L are from previous daily close, high, low
    df_1d = get_htf_data(prices, '1d')  # reload for OHLC (small overhead, called once)
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get previous day's OHLC for each 4h bar
    # align_htf_to_ltf with additional_delay_bars=1 to get previous completed day
    prev_close_1d = align_htf_to_ltf(prices, df_1d, df_1d['close'].values, additional_delay_bars=1)
    prev_high_1d = align_htf_to_ltf(prices, df_1d, df_1d['high'].values, additional_delay_bars=1)
    prev_low_1d = align_htf_to_ltf(prices, df_1d, df_1d['low'].values, additional_delay_bars=1)
    
    # Camarilla R3 and S3
    camarilla_r3 = prev_close_1d + (prev_high_1d - prev_low_1d) * 1.1 / 2
    camarilla_s3 = prev_close_1d - (prev_high_1d - prev_low_1d) * 1.1 / 2
    # Also calculate R2 and S2 for exits
    camarilla_r2 = prev_close_1d + (prev_high_1d - prev_low_1d) * 1.1 / 4
    camarilla_s2 = prev_close_1d - (prev_high_1d - prev_low_1d) * 1.1 / 4
    
    # 20-period volume average for spike filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # EMA34 (34), volume MA (20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_aligned[i]) or np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(camarilla_r2[i]) or np.isnan(camarilla_s2[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ema_val = ema_34_aligned[i]
        r3 = camarilla_r3[i]
        s3 = camarilla_s3[i]
        r2 = camarilla_r2[i]
        s2 = camarilla_s2[i]
        vol_ma_val = vol_ma[i]
        
        # Calculate EMA34 slope for trend direction (rising/falling)
        if i >= start_idx + 1:
            ema_prev = ema_34_aligned[i-1]
            ema_rising = ema_val > ema_prev
            ema_falling = ema_val < ema_prev
        else:
            ema_rising = False
            ema_falling = False
        
        if position == 0:
            # Long: Break above Camarilla R3 AND EMA34 rising AND volume spike
            if price > r3 and ema_rising and volume[i] > 2.0 * vol_ma_val:
                signals[i] = 0.25
                position = 1
            # Short: Break below Camarilla S3 AND EMA34 falling AND volume spike
            elif price < s3 and ema_falling and volume[i] > 2.0 * vol_ma_val:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: price touches Camarilla S2 OR EMA34 starts falling
                if price < s2 or (i >= start_idx + 1 and ema_val < ema_34_aligned[i-1]):
                    exit_signal = True
            elif position == -1:
                # Short exit: price touches Camarilla R2 OR EMA34 starts rising
                if price > r2 or (i >= start_idx + 1 and ema_val > ema_34_aligned[i-1]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0