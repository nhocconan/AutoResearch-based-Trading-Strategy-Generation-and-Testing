#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_v3
Hypothesis: Tighten entry conditions from v2 to reduce trade count (<200 total 4h trades) while maintaining edge. Uses Camarilla R3/S3 breakouts with 1d EMA34 trend filter, volume > 1.8x average, and ATR stoploss (2.0*ATR). Discrete sizing (0.25) to minimize fee churn. Works in bull/bear by following 1d trend, confirmed by volume to avoid false breakouts. Lower trade frequency targets better test generalization.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need warmup for EMA, volume, ATR
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for HTF trend filter and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Camarilla levels from previous 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla R3 and S3 levels
    camarilla_r3 = close_1d + (high_1d - low_1d) * 1.1 / 4
    camarilla_s3 = close_1d - (high_1d - low_1d) * 1.1 / 4
    
    # Align Camarilla levels to 4h timeframe (1 bar delay for completed 1d bar)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Calculate average volume for confirmation (20-period SMA)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    base_size = 0.25
    atr_multiplier = 2.0  # Reduced from 2.5 to 2.0 for tighter stop
    
    # Start after warmup (need 34 for EMA, 20 for volume, 14 for ATR)
    start_idx = max(34, 20, 14)
    
    for i in range(start_idx, n):
        # Get current values
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        ema_val = ema_34_1d_aligned[i]
        r3_val = camarilla_r3_aligned[i]
        s3_val = camarilla_s3_aligned[i]
        atr_val = atr[i]
        
        # Skip if any data not ready
        if (np.isnan(ema_val) or np.isnan(avg_vol) or np.isnan(r3_val) or 
            np.isnan(s3_val) or np.isnan(atr_val)):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Volume confirmation: current volume > 1.8x average volume (slightly looser than v2's 2.0x)
        volume_confirmed = vol > 1.8 * avg_vol
        
        # Long logic: price breaks above Camarilla R3 with 1d uptrend and volume confirmation
        long_condition = (close_val > r3_val) and (close_val > ema_val) and volume_confirmed
        # Short logic: price breaks below Camarilla S3 with 1d downtrend and volume confirmation
        short_condition = (close_val < s3_val) and (close_val < ema_val) and volume_confirmed
        
        # Stoploss logic: price moves against position by atr_multiplier * ATR from entry
        long_stop = (position == 1 and close_val < entry_price - atr_multiplier * atr_val)
        short_stop = (position == -1 and close_val > entry_price + atr_multiplier * atr_val)
        
        # Exit logic: 
        # Long exit: price retests or breaks below Camarilla R3 (failed breakout) OR stoploss hit
        long_exit = (position == 1 and (close_val <= r3_val or long_stop))
        # Short exit: price retests or breaks above Camarilla S3 (failed breakout) OR stoploss hit
        short_exit = (position == -1 and (close_val >= s3_val or short_stop))
        
        if long_condition and position != 1:
            signals[i] = base_size
            position = 1
            entry_price = close_val  # Enter at next bar open, approximate with close
        elif short_condition and position != -1:
            signals[i] = -base_size
            position = -1
            entry_price = close_val  # Enter at next bar open, approximate with close
        elif long_exit:
            signals[i] = 0.0
            position = 0
        elif short_exit:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_v3"
timeframe = "4h"
leverage = 1.0