#!/usr/bin/env python3
"""
1h_Camarilla_R3S3_Breakout_4hTrend_VolumeSpike
Hypothesis: On 1h timeframe, use 4h Camarilla R3/S3 breakout with 1d EMA50 trend filter and volume confirmation (>2x average volume) for signal direction, while 1h provides precise entry timing. Discrete position sizing (0.20) minimizes fee churn. Target: 60-150 total trades over 4 years (15-37/year) on 1h timeframe. Works in both bull and bear markets via 1d trend alignment and strict volume confirmation. Session filter (08-20 UTC) reduces noise trades.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need warmup for EMA and ATR
        return np.zeros(n)
    
    # Pre-compute session hours to avoid datetime operations in loop
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 4h data for HTF Camarilla levels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Load 1d data for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar has no previous close
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate average volume for confirmation (20-period SMA)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    base_size = 0.20
    atr_multiplier = 2.5  # ATR stoploss multiplier
    
    # Start after warmup (need 20 for Camarilla, 50 for EMA, 14 for ATR)
    start_idx = max(20, 50, 14)
    
    for i in range(start_idx, n):
        # Session filter: 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            # Outside session: flatten position
            signals[i] = 0.0
            position = 0
            continue
        
        # Need previous day's OHLC for Camarilla levels (using 4h data)
        if i < 1:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
            
        # Previous period's high, low, close (for Camarilla calculation) from 4h data
        # We need to get the previous completed 4h bar's OHLC
        # Since we're on 1h timeframe, we look back 4 bars for previous 4h bar
        if i < 4:
            # Hold current position if not enough 4h history
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
            
        # Get 4h aligned arrays for Camarilla calculation
        # We'll use the 4h data aligned to 1h timeframe
        df_4h_close = get_htf_data(prices, '4h')['close'].values
        df_4h_high = get_htf_data(prices, '4h')['high'].values
        df_4h_low = get_htf_data(prices, '4h')['low'].values
        
        # Align 4h OHLC to 1h timeframe
        h_close_4h_aligned = align_htf_to_ltf(prices, df_4h, df_4h_close)
        h_high_4h_aligned = align_htf_to_ltf(prices, df_4h, df_4h_high)
        h_low_4h_aligned = align_htf_to_ltf(prices, df_4h, df_4h_low)
        
        # Get previous completed 4h bar's OHLC (current 1h bar index i corresponds to some 4h bar)
        # We need the 4h bar that completed before the current 1h bar
        # Since 4h = 4 * 1h, we look at index i-4 for the previous 4h bar's close
        prev_high = h_high_4h_aligned[i-4] if i-4 >= 0 else h_high_4h_aligned[0]
        prev_low = h_low_4h_aligned[i-4] if i-4 >= 0 else h_low_4h_aligned[0]
        prev_close = h_close_4h_aligned[i-4] if i-4 >= 0 else h_close_4h_aligned[0]
        
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
        ema_val = ema_50_1d_aligned[i]
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
        
        # Volume confirmation: current volume > 2.0x average volume (stricter for fewer trades)
        volume_confirmed = vol > 2.0 * avg_vol
        
        # Long logic: price breaks above R3 with 1d uptrend and volume confirmation
        long_condition = (close_val > r3) and (close_val > ema_val) and volume_confirmed
        # Short logic: price breaks below S3 with 1d downtrend and volume confirmation
        short_condition = (close_val < s3) and (close_val < ema_val) and volume_confirmed
        
        # Exit logic: trend reversal (close crosses 1d EMA50)
        exit_long = close_val < ema_val
        exit_short = close_val > ema_val
        
        # ATR-based stoploss
        if position == 1:
            stop_price = entry_price - atr_multiplier * atr_val
            if close_val < stop_price:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:
            stop_price = entry_price + atr_multiplier * atr_val
            if close_val > stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        if long_condition and position != 1:
            signals[i] = base_size
            position = 1
            entry_price = close_val  # Enter at next bar open, approximate with close
        elif short_condition and position != -1:
            signals[i] = -base_size
            position = -1
            entry_price = close_val  # Enter at next bar open, approximate with close
        elif position == 1 and exit_long:
            signals[i] = 0.0
            position = 0
        elif position == -1 and exit_short:
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

name = "1h_Camarilla_R3S3_Breakout_4hTrend_VolumeSpike"
timeframe = "1h"
leverage = 1.0