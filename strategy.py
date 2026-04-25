#!/usr/bin/env python3
"""
12h Camarilla H3/L3 Breakout + 1d EMA50 Trend + Volume Spike
Hypothesis: Camarilla pivot levels (H3/L3) act as strong support/resistance on 12h timeframe. 
Breakouts above H3 (long) or below L3 (short) with volume confirmation (>2x 20-period MA) 
and trend alignment (price vs 1d EMA50) capture momentum moves. 
ATR-based stoploss manages risk. Designed for 12h timeframe with 50-150 total trades over 4 years.
Works in both bull and bear markets via trend filter and volume confirmation.
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
    
    # Calculate Camarilla pivot levels (based on previous day's OHLC)
    # Need to get daily OHLC from 1d data
    camarilla_H3 = np.full(n, np.nan)
    camarilla_L3 = np.full(n, np.nan)
    
    # For each 12h bar, use previous 1d bar's OHLC to calculate Camarilla levels
    for i in range(len(prices)):
        # Get the index of the most recent completed 1d bar
        # Since prices is 12h, we need to map to daily index
        # We'll use the aligned 1d data - for simplicity, we use the previous 1d bar's values
        if i >= 2:  # Need at least 2 bars to have previous day
            # Use previous 1d bar's OHLC (aligned to current 12h bar)
            # We'll approximate by using the 1d data that's been aligned
            # Actually, we need to get the actual 1d OHLC for the previous day
            # Let's get the 1d OHLC arrays
            pass  # We'll calculate this properly below
    
    # Better approach: calculate Camarilla levels from 1d OHLC, then align to 12h
    if len(df_1d) >= 2:
        # Calculate Camarilla levels for each 1d bar
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        close_1d = df_1d['close'].values
        
        camarilla_H3_1d = np.full(len(df_1d), np.nan)
        camarilla_L3_1d = np.full(len(df_1d), np.nan)
        
        for i in range(1, len(df_1d)):  # Start from 1 to have previous day
            # Camarilla levels based on previous day's OHLC
            # H3 = close + 1.1 * (high - low) / 4
            # L3 = close - 1.1 * (high - low) / 4
            prev_high = high_1d[i-1]
            prev_low = low_1d[i-1]
            prev_close = close_1d[i-1]
            range_val = prev_high - prev_low
            
            camarilla_H3_1d[i] = prev_close + 1.1 * range_val / 4
            camarilla_L3_1d[i] = prev_close - 1.1 * range_val / 4
        
        # Align Camarilla levels from 1d to 12h timeframe
        camarilla_H3 = align_htf_to_ltf(prices, df_1d, camarilla_H3_1d)
        camarilla_L3 = align_htf_to_ltf(prices, df_1d, camarilla_L3_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for EMA50, volume MA, ATR, and Camarilla
    start_idx = max(50, 20, 14, 2)  # 2 for Camarilla (need previous day)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(atr[i]) or np.isnan(camarilla_H3[i]) or np.isnan(camarilla_L3[i])):
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
        camarilla_H3_val = camarilla_H3[i]
        camarilla_L3_val = camarilla_L3[i]
        
        # Trend filter: price relative to 1d EMA50
        uptrend = curr_close > ema_50_val
        downtrend = curr_close < ema_50_val
        
        # Volume confirmation: current volume > 2.0 * 20-period average
        volume_confirm = curr_volume > 2.0 * vol_ma
        
        if position == 0:
            # Look for breakout signals
            # Long: price breaks above Camarilla H3 with volume confirmation in uptrend
            long_breakout = (curr_close > camarilla_H3_val) and volume_confirm and uptrend
            # Short: price breaks below Camarilla L3 with volume confirmation in downtrend
            short_breakout = (curr_close < camarilla_L3_val) and volume_confirm and downtrend
            
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
            # Long position management
            # Stoploss: 2 * ATR below entry
            stop_loss = entry_price - 2.0 * atr_val
            # Exit conditions: price closes below Camarilla L3 OR stoploss hit OR EMA50 trend turns down
            if curr_close < camarilla_L3_val or curr_close < stop_loss or curr_close < ema_50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Stoploss: 2 * ATR above entry
            stop_loss = entry_price + 2.0 * atr_val
            # Exit conditions: price closes above Camarilla H3 OR stoploss hit OR EMA50 trend turns up
            if curr_close > camarilla_H3_val or curr_close > stop_loss or curr_close > ema_50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_H3L3_Breakout_1dEMA50_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0