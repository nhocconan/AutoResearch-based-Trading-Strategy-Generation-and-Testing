#!/usr/bin/env python3
"""
4h Camarilla R3/S3 Breakout + 1d EMA50 Trend + Volume Spike + ATR Stoploss
Hypothesis: Camarilla pivot levels (R3/S3) act as strong support/resistance on 4h charts. Breakouts above R3 or below S3 with volume confirmation (>1.5x 20-period vol MA) and 1d EMA50 trend filter capture institutional flow. ATR-based stoploss (2x ATR) manages risk. Designed for 4h timeframe targeting 75-200 total trades over 4 years, effective in both bull and bear markets via trend alignment and volatility-based entries.
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
    
    # Calculate 20-period volume MA for volume spike confirmation (4h)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    # Calculate ATR(14) for stoploss (4h)
    atr = np.full(n, np.nan)
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    for i in range(14, n):
        atr[i] = np.mean(tr[i-13:i+1])
    
    # Calculate Camarilla levels (R3, S3) from previous day (4h)
    camarilla_R3 = np.full(n, np.nan)
    camarilla_S3 = np.full(n, np.nan)
    for i in range(1, n):
        # Use previous day's high, low, close (1d data) to calculate today's Camarilla
        # We need to get the 1d OHLC for the previous completed day
        # Since we're on 4h timeframe, we'll use the 1d data from get_htf_data
        pass  # Will calculate below using 1d data
    
    # Calculate Camarilla levels using 1d OHLC
    # Camarilla: R3 = close + 1.1*(high-low)*1.1/4, S3 = close - 1.1*(high-low)*1.1/4
    # Actually standard Camarilla: R4 = close + 1.1*(high-low)*1.1/2, R3 = close + 1.1*(high-low)*1.1/4
    # Simplified: R3 = close + 1.1*(high-low)*1.1/4, S3 = close - 1.1*(high-low)*1.1/4
    # We'll calculate from 1d data and align to 4h
    if len(df_1d) >= 2:
        # Calculate Camarilla for each 1d bar using that day's OHLC
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        close_1d_arr = df_1d['close'].values
        
        camarilla_R3_1d = close_1d_arr + 1.1 * (high_1d - low_1d) * 1.1 / 4
        camarilla_S3_1d = close_1d_arr - 1.1 * (high_1d - low_1d) * 1.1 / 4
        
        # Align to 4h timeframe
        camarilla_R3 = align_htf_to_ltf(prices, df_1d, camarilla_R3_1d)
        camarilla_S3 = align_htf_to_ltf(prices, df_1d, camarilla_S3_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for EMA50, volume MA, ATR, and Camarilla
    start_idx = max(50, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(atr[i]) or np.isnan(camarilla_R3[i]) or np.isnan(camarilla_S3[i])):
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
        camarilla_R3_val = camarilla_R3[i]
        camarilla_S3_val = camarilla_S3[i]
        
        # Trend filter: price relative to 1d EMA50
        uptrend = curr_close > ema_50_val
        downtrend = curr_close < ema_50_val
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        volume_confirm = curr_volume > 1.5 * vol_ma
        
        if position == 0:
            # Look for breakout signals
            # Long: price breaks above Camarilla R3 with volume confirmation in uptrend
            long_breakout = (curr_close > camarilla_R3_val) and volume_confirm and uptrend
            # Short: price breaks below Camarilla S3 with volume confirmation in downtrend
            short_breakout = (curr_close < camarilla_S3_val) and volume_confirm and downtrend
            
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
            # Exit conditions: price closes below Camarilla S3 OR stoploss hit OR EMA50 trend turns down
            if curr_close < camarilla_S3_val or curr_close < stop_loss or curr_close < ema_50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Stoploss: 2 * ATR above entry
            stop_loss = entry_price + 2.0 * atr_val
            # Exit conditions: price closes above Camarilla R3 OR stoploss hit OR EMA50 trend turns up
            if curr_close > camarilla_R3_val or curr_close > stop_loss or curr_close > ema_50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R3S3_Breakout_1dEMA50_Trend_VolumeSpike_ATRStop"
timeframe = "4h"
leverage = 1.0