#!/usr/bin/env python3
"""
12h Williams Alligator + 1d EMA50 Trend + Volume Spike
Hypothesis: Williams Alligator (jaw/teeth/lips) identifies trend absence (all lines entwined) vs presence (lines separated). 
Breakouts occur when price crosses lips with jaw/teeth aligned + volume confirmation (>2x 20-period volume MA). 
1d EMA50 ensures alignment with daily trend to avoid counter-trend trades. 
Designed for 12h timeframe targeting 50-150 total trades over 4 years. 
Works in both bull and bear markets via daily trend filter and volume confirmation to reduce false breakouts.
Alligator uses SMAs (5,8,13) with specific shifts as per Bill Williams.
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
    
    # Get 1d data for EMA50 trend (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need at least 50 days for EMA50
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = pd.Series(df_1d['close'])
    ema_50_1d = close_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 20-period volume MA for volume spike confirmation (12h)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    # Calculate ATR(14) for stoploss (12h)
    atr = np.full(n, np.nan)
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    for i in range(14, n):
        atr[i] = np.mean(tr[i-13:i+1])
    
    # Williams Alligator on 12h: Jaw (13,8), Teeth (8,5), Lips (5,3) - all SMAs with specific shifts
    # Jaw: SMA(13) shifted 8 bars forward
    # Teeth: SMA(8) shifted 5 bars forward  
    # Lips: SMA(5) shifted 3 bars forward
    jaw = np.full(n, np.nan)
    teeth = np.full(n, np.nan)
    lips = np.full(n, np.nan)
    
    # Calculate SMAs
    sma_5 = np.full(n, np.nan)
    sma_8 = np.full(n, np.nan)
    sma_13 = np.full(n, np.nan)
    
    for i in range(4, n):  # SMA(5) needs 5 bars
        sma_5[i] = np.mean(close[i-4:i+1])
    for i in range(7, n):  # SMA(8) needs 8 bars
        sma_8[i] = np.mean(close[i-7:i+1])
    for i in range(12, n):  # SMA(13) needs 13 bars
        sma_13[i] = np.mean(close[i-12:i+1])
    
    # Apply shifts: Jaw (SMA13 shifted 8), Teeth (SMA8 shifted 5), Lips (SMA5 shifted 3)
    for i in range(8, n):
        jaw[i] = sma_13[i-8] if i >= 8+12 else np.nan  # SMA13 needs 12 lookback, shifted 8
    for i in range(5, n):
        teeth[i] = sma_8[i-5] if i >= 5+7 else np.nan   # SMA8 needs 7 lookback, shifted 5
    for i in range(3, n):
        lips[i] = sma_5[i-3] if i >= 3+4 else np.nan    # SMA5 needs 4 lookback, shifted 3
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for EMA50, volume MA, ATR, and Alligator lines
    start_idx = max(50, 20, 14, 13, 8, 5, 3)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(atr[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_50_val = ema_50_1d_aligned[i]
        vol_ma = vol_ma_20[i]
        atr_val = atr[i]
        jaw_val = jaw[i]
        teeth_val = teeth[i]
        lips_val = lips[i]
        
        # Trend filter: price relative to 1d EMA50
        uptrend = curr_close > ema_50_val
        downtrend = curr_close < ema_50_val
        
        # Volume confirmation: current volume > 2.0 * 20-period average
        volume_confirm = curr_volume > 2.0 * vol_ma
        
        # Alligator conditions: 
        # Trending market: lines are separated (jaw < teeth < lips for uptrend OR jaw > teeth > lips for downtrend)
        # Merging market: lines are entwined (no clear separation)
        alligator_uptrend = jaw_val < teeth_val < lips_val  # Mouth opening up
        alligator_downtrend = jaw_val > teeth_val > lips_val  # Mouth opening down
        alligator_sleeping = not (alligator_uptrend or alligator_downtrend)  # Lines entwined
        
        if position == 0:
            # Look for breakout signals when Alligator wakes up (lines separating) + price crosses lips
            # Long: price crosses above lips with uptrend alignment + volume confirmation
            long_signal = (curr_close > lips_val) and alligator_uptrend and volume_confirm and uptrend
            # Short: price crosses below lips with downtrend alignment + volume confirmation
            short_signal = (curr_close < lips_val) and alligator_downtrend and volume_confirm and downtrend
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Stoploss: 2 * ATR below entry
            stop_loss = entry_price - 2.0 * atr_val
            # Exit conditions: price closes below lips OR stoploss hit OR Alligator starts sleeping (loss of momentum) OR EMA50 trend turns down
            if curr_close < lips_val or curr_close < stop_loss or alligator_sleeping or curr_close < ema_50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Stoploss: 2 * ATR above entry
            stop_loss = entry_price + 2.0 * atr_val
            # Exit conditions: price closes above lips OR stoploss hit OR Alligator starts sleeping OR EMA50 trend turns up
            if curr_close > lips_val or curr_close > stop_loss or alligator_sleeping or curr_close > ema_50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Williams_Alligator_1dEMA50_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0