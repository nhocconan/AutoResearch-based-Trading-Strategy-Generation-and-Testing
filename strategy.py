#!/usr/bin/env python3
"""
4h_Camarilla_R3S3_Breakout_1dEMA34_ATRStop
Hypothesis: Camarilla R3/S3 breakout on 4h with 1d EMA34 trend filter and ATR-based stoploss for risk management.
Uses tighter volume confirmation (>2.5x average) and discrete position sizing (0.25) to reduce trade frequency.
ATR stoploss placed at 2.0x ATR below entry for longs, above entry for shorts.
Designed to work in both bull and bear markets by requiring 1d trend alignment and strong breakout levels.
Target: 50-120 trades over 4 years (12-30/year) on 4h timeframe to avoid fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need warmup for EMA and ATR
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate average volume for confirmation (20-period SMA)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate ATR (14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    base_size = 0.25
    
    # Start after warmup (need 20 for Camarilla calculation, 34 for EMA, 14 for ATR)
    start_idx = max(20, 34, 14)
    
    for i in range(start_idx, n):
        # Need previous day's OHLC for Camarilla levels
        if i < 1:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
            
        # Previous period's high, low, close (for Camarilla calculation)
        prev_high = high[i-1]
        prev_low = low[i-1]
        prev_close = close[i-1]
        
        # Calculate Camarilla levels
        range_val = prev_high - prev_low
        if range_val <= 0:
            # Hold current position if invalid range
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
            
        # Camarilla R3 and S3 levels (stronger levels)
        r3 = prev_close + range_val * 1.1 / 4
        s3 = prev_close - range_val * 1.1 / 4
        
        close_val = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        ema_val = ema_34_1d_aligned[i]
        atr_val = atr[i]
        
        # Skip if any data not ready
        if np.isnan(r3) or np.isnan(s3) or np.isnan(ema_val) or np.isnan(avg_vol) or np.isnan(atr_val):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Volume confirmation: current volume > 2.5x average volume (stricter for fewer trades)
        volume_confirmed = vol > 2.5 * avg_vol
        
        # Long logic: price breaks above R3 with 1d uptrend and volume confirmation
        long_condition = (close_val > r3) and (close_val > ema_val) and volume_confirmed
        # Short logic: price breaks below S3 with 1d downtrend and volume confirmation
        short_condition = (close_val < s3) and (close_val < ema_val) and volume_confirmed
        
        # Stoploss logic: ATR-based stop
        stop_long = position == 1 and close_val < (entry_price - 2.0 * atr_val)
        stop_short = position == -1 and close_val > (entry_price + 2.0 * atr_val)
        
        # Exit logic: trend reversal or opposite Camarilla level break
        exit_long = (close_val < ema_val) or (close_val < s3)
        exit_short = (close_val > ema_val) or (close_val > r3)
        
        if long_condition and position != 1:
            signals[i] = base_size
            position = 1
            entry_price = close_val
        elif short_condition and position != -1:
            signals[i] = -base_size
            position = -1
            entry_price = close_val
        elif position == 1 and (exit_long or stop_long):
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
        elif position == -1 and (exit_short or stop_short):
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_ATRStop"
timeframe = "4h"
leverage = 1.0