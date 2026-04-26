#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_1d_EMA34_Trend_VolumeSpike_v2
Hypothesis: Trade Camarilla R3/S3 breakouts on 4h only when aligned with 1d EMA34 trend and confirmed by volume spike (>2.0x 20-bar average). Uses ATR trailing stop. Optimized for fewer trades (<150 total) by tightening volume confirmation and adding choppiness regime filter to avoid whipsaws in ranging markets. Works in both bull (long at R3 breakout) and bear (short at S3 breakdown) markets.
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
    
    # Get 1d data for EMA34 trend filter and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate EMA34 on 1d for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from previous 1d bar (HLC of daily)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_arr = df_1d['close'].values
    
    # Calculate Camarilla R3 and S3 for each 1d bar
    camarilla_r3_1d = close_1d_arr + ((high_1d - low_1d) * 1.1 / 4)
    camarilla_s3_1d = close_1d_arr - ((high_1d - low_1d) * 1.1 / 4)
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3_1d)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3_1d)
    
    # ATR for stoploss and volatility filter
    atr_period = 14
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=atr_period, min_periods=atr_period, adjust=False).mean().values
    
    # Volume spike: current volume > 2.2 * 20-period average (tighter)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.2 * vol_ma)
    
    # Choppiness regime filter: avoid trading in high chop (>61.8) to reduce whipsaws
    chop_period = 14
    true_range = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    true_range[0] = high[0] - low[0]
    atr_chop = pd.Series(true_range).rolling(window=chop_period, min_periods=chop_period).sum().values
    max_high = pd.Series(high).rolling(window=chop_period, min_periods=chop_period).max().values
    min_low = pd.Series(low).rolling(window=chop_period, min_periods=chop_period).min().values
    chop = 100 * np.log10(atr_chop / (np.maximum(max_high - min_low, 1e-10))) / np.log10(chop_period)
    chop_filter = chop < 61.8  # Only trade when NOT in high chop regime
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup: max of EMA34 (34), ATR (14), volume MA (20), chop (14)
    start_idx = max(34, 14, 20, 14) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(atr[i]) or
            np.isnan(vol_ma[i]) or
            np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(chop[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        ema_34_val = ema_34_1d_aligned[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        atr_val = atr[i]
        vol_spike = volume_spike[i]
        r3_level = camarilla_r3_aligned[i]
        s3_level = camarilla_s3_aligned[i]
        chop_ok = chop_filter[i]
        
        if position == 0:
            # Long: Break above R3, above 1d EMA34, with volume spike, NOT in high chop
            long_signal = (high_val > r3_level) and (close_val > ema_34_val) and vol_spike and chop_ok
            
            # Short: Break below S3, below 1d EMA34, with volume spike, NOT in high chop
            short_signal = (low_val < s3_level) and (close_val < ema_34_val) and vol_spike and chop_ok
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
                highest_since_entry = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
                lowest_since_entry = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            highest_since_entry = max(highest_since_entry, close_val)
            # Exit: Close below EMA34 (trend change) OR trailing stop (2.0*ATR below high - tighter)
            if (close_val < ema_34_val) or (close_val < highest_since_entry - 2.0 * atr_val):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            lowest_since_entry = min(lowest_since_entry, close_val)
            # Exit: Close above EMA34 (trend change) OR trailing stop (2.0*ATR above low - tighter)
            if (close_val > ema_34_val) or (close_val > lowest_since_entry + 2.0 * atr_val):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R3_S3_Breakout_1d_EMA34_Trend_VolumeSpike_v2"
timeframe = "4h"
leverage = 1.0