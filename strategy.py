#!/usr/bin/env python3
"""
6h Williams Alligator + 1d EMA50 Trend with Volume Confirmation
Hypothesis: Williams Alligator (JAW/TEETH/LIPS) identifies trending vs ranging markets on 6h.
In trending markets (JAW > TEETH > LIPS for uptrend, reverse for downtrend), we take breakout entries
in the direction of the 1d EMA50 trend with volume confirmation. In ranging markets, we stay flat.
This avoids whipsaws in chop while capturing strong trends. Works in bull via long entries and
bear via short entries. Uses ATR-based trailing stop for risk control.
Target: 50-150 total trades over 4 years (12-37/year) on 6h timeframe.
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
    
    # Get 1d data for EMA50 trend filter (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams Alligator on 6h: SMAs with specific periods and shifts
    # JAW: 13-period SMMA shifted 8 bars
    # TEETH: 8-period SMMA shifted 5 bars
    # LIPS: 5-period SMMA shifted 3 bars
    # SMMA = smoothed moving average (RMA/Wilder's EMA)
    def smma(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        result = np.full(len(arr), np.nan)
        # First value is SMA
        result[period-1] = np.mean(arr[:period])
        # Wilder's EMA: today = (yesterday * (period-1) + today) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    # Apply shifts (JAW: +8, TEETH: +5, LIPS: +3)
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    # First shifted values are invalid
    jaw_shifted[:8] = np.nan
    teeth_shifted[:5] = np.nan
    lips_shifted[:3] = np.nan
    
    # Calculate ATR(14) for stoploss
    atr_14 = np.full(n, np.nan)
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    for i in range(14, n):
        atr_14[i] = np.mean(tr[i-13:i+1])
    
    # Calculate 20-period volume MA for volume confirmation
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    atr_stop = 0.0
    
    # Start index: need enough for Alligator, EMA50_1d, volume MA, ATR to propagate
    start_idx = max(50, 20, 14) + 8  # +8 for jaw shift
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(jaw_shifted[i]) or 
            np.isnan(teeth_shifted[i]) or 
            np.isnan(lips_shifted[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i]) or 
            np.isnan(atr_14[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        jaw_val = jaw_shifted[i]
        teeth_val = teeth_shifted[i]
        lips_val = lips_shifted[i]
        ema50_1d = ema_50_1d_aligned[i]
        vol_ma = vol_ma_20[i]
        atr = atr_14[i]
        
        # Alligator trend detection
        # Uptrend: JAW > TEETH > LIPS (alligator mouth open up)
        # Downtrend: JAW < TEETH < LIPS (alligator mouth open down)
        # Ranging: otherwise (alligator sleeping)
        is_uptrend = (jaw_val > teeth_val) and (teeth_val > lips_val)
        is_downtrend = (jaw_val < teeth_val) and (teeth_val < lips_val)
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        volume_confirm = curr_volume > 1.5 * vol_ma
        
        if position == 0:
            # Long entry: Alligator uptrend + price above LIPS + 1d EMA50 uptrend + volume
            long_entry = is_uptrend and (curr_close > lips_val) and (curr_close > ema50_1d) and volume_confirm
            # Short entry: Alligator downtrend + price below LIPS + 1d EMA50 downtrend + volume
            short_entry = is_downtrend and (curr_close < lips_val) and (curr_close < ema50_1d) and volume_confirm
            
            if long_entry:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                atr_stop = curr_close - 2.0 * atr  # Initial stop
            elif short_entry:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
                atr_stop = curr_close + 2.0 * atr  # Initial stop
        elif position == 1:
            # Update trailing stop: raise stop to highest high - 2.0*ATR
            # Track highest high since entry
            if i == start_idx or position == 0:  # reset on new trade
                highest_since_entry = curr_high
            else:
                highest_since_entry = max(getattr(generate_signals, 'highest_since_entry', curr_high), curr_high)
            generate_signals.highest_since_entry = highest_since_entry
            atr_stop = max(atr_stop, highest_since_entry - 2.0 * atr)
            # Exit long: price closes below trailing stop
            if curr_close < atr_stop:
                signals[i] = 0.0
                position = 0
                if hasattr(generate_signals, 'highest_since_entry'):
                    delattr(generate_signals, 'highest_since_entry')
            else:
                signals[i] = 0.25
        elif position == -1:
            # Update trailing stop: lower stop to lowest low + 2.0*ATR
            # Track lowest low since entry
            if i == start_idx or position == 0:  # reset on new trade
                lowest_since_entry = curr_low
            else:
                lowest_since_entry = min(getattr(generate_signals, 'lowest_since_entry', curr_low), curr_low)
            generate_signals.lowest_since_entry = lowest_since_entry
            atr_stop = min(atr_stop, lowest_since_entry + 2.0 * atr)
            # Exit short: price closes above trailing stop
            if curr_close > atr_stop:
                signals[i] = 0.0
                position = 0
                if hasattr(generate_signals, 'lowest_since_entry'):
                    delattr(generate_signals, 'lowest_since_entry')
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Williams_Alligator_1dEMA50_Trend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0