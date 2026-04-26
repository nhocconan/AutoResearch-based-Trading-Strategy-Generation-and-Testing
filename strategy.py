#!/usr/bin/env python3
"""
4h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike_v1
Hypothesis: Camarilla R3/S3 breakouts on 4h with 1d EMA50 trend filter and volume spike (>2x average) capture strong institutional moves. R3/S3 levels offer wider bands than R1/S1, reducing false breakouts while maintaining trend alignment. Volume confirmation ensures momentum. Designed for 4h to target 20-50 trades/year with discrete sizing (0.25) and ATR-based stoploss.
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
    
    # Load 1d data ONCE before loop for Camarilla levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Camarilla levels from previous 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_r3 = close_1d + (high_1d - low_1d) * 1.1 / 4
    camarilla_s3 = close_1d - (high_1d - low_1d) * 1.1 / 4
    
    # Align to 1d (wait for completed 1d bar)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Average volume for confirmation (24-period SMA = 6h * 4 = 24 periods on 4h)
    avg_volume = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    base_size = 0.25
    
    # Warmup: max of EMA(50), volume(24), ATR(14)
    start_idx = max(50, 24, 14)
    
    for i in range(start_idx, n):
        close_val = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        ema_val = ema_50_1d_aligned[i]
        r3_val = camarilla_r3_aligned[i]
        s3_val = camarilla_s3_aligned[i]
        atr_val = atr[i]
        
        # Skip if any data not ready
        if (np.isnan(ema_val) or np.isnan(avg_vol) or np.isnan(r3_val) or 
            np.isnan(s3_val) or np.isnan(atr_val)):
            # Hold current position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        # Volume confirmation: current volume > 2.0x average volume
        volume_confirmed = vol > 2.0 * avg_vol
        
        # Trend filter: price vs 1d EMA50
        uptrend = close_val > ema_val
        downtrend = close_val < ema_val
        
        # Long: price CLOSES above R3 with 1d uptrend and volume
        long_condition = (close_val > r3_val) and uptrend and volume_confirmed
        # Short: price CLOSES below S3 with 1d downtrend and volume
        short_condition = (close_val < s3_val) and downtrend and volume_confirmed
        
        # Stoploss: 2 * ATR from entry
        long_stop = position == 1 and entry_price > 0 and close_val < (entry_price - 2.0 * atr_val)
        short_stop = position == -1 and entry_price > 0 and close_val > (entry_price + 2.0 * atr_val)
        
        # Exit: price retests broken level (optional early exit)
        long_exit = (position == 1 and close_val <= r3_val)
        short_exit = (position == -1 and close_val >= s3_val)
        
        if (long_condition or short_condition) and position == 0:
            if long_condition:
                signals[i] = base_size
                position = 1
                entry_price = close_val
            else:  # short_condition
                signals[i] = -base_size
                position = -1
                entry_price = close_val
        elif long_stop or short_stop or long_exit or short_exit:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
        else:
            # Hold position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
    
    return signals

name = "4h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0