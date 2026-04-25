#!/usr/bin/env python3
"""
1h Camarilla Pivot Breakout + 4h EMA20 Trend + Volume Spike
Hypothesis: 1h Camarilla H3/L3 breakouts capture intraday momentum. 4h EMA20 filter ensures alignment with higher timeframe trend. Volume confirmation (>1.5x 20-period MA) filters false breakouts. Session filter (08-20 UTC) reduces noise. Designed for 1h timeframe with 60-150 total trades over 4 years, working in both bull and bear markets via trend filter and volume confirmation.
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
    
    # Get 4h data for EMA20 trend (call ONCE before loop)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:  # Need at least 20 periods for EMA20
        return np.zeros(n)
    
    # Calculate 4h EMA20 for trend filter
    close_4h = pd.Series(df_4h['close'])
    ema_20_4h = close_4h.ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # Calculate 20-period volume MA for volume spike confirmation (1h)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    # Calculate ATR(14) for stoploss (1h)
    atr = np.full(n, np.nan)
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    for i in range(14, n):
        atr[i] = np.mean(tr[i-13:i+1])
    
    # Calculate Camarilla pivots (based on previous day's range) (1h)
    camarilla_h3 = np.full(n, np.nan)
    camarilla_l3 = np.full(n, np.nan)
    camarilla_h4 = np.full(n, np.nan)
    camarilla_l4 = np.full(n, np.nan)
    
    # Calculate daily high/low/close for pivot calculation
    # We'll use rolling window of 24 periods (24*1h = 1 day) as approximation
    # For better accuracy, we could load 1d data, but this keeps it simple and avoids look-ahead
    for i in range(24, n):
        # Previous day's high, low, close (24 periods ago)
        prev_high = np.max(high[i-24:i])
        prev_low = np.min(low[i-24:i])
        prev_close = close[i-1]  # Previous period close
        
        # Calculate pivot point
        pivot = (prev_high + prev_low + prev_close) / 3.0
        
        # Calculate Camarilla levels
        range_val = prev_high - prev_low
        camarilla_h3[i] = pivot + range_val * 1.1 / 4
        camarilla_l3[i] = pivot - range_val * 1.1 / 4
        camarilla_h4[i] = pivot + range_val * 1.1 / 2
        camarilla_l4[i] = pivot - range_val * 1.1 / 2
    
    # Session filter: 08-20 UTC (pre-compute hours)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for EMA20, volume MA, ATR, and Camarilla
    start_idx = max(20, 20, 14, 24)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_20_4h_aligned[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(atr[i]) or np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: only trade between 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_20_val = ema_20_4h_aligned[i]
        vol_ma = vol_ma_20[i]
        atr_val = atr[i]
        camarilla_h3_val = camarilla_h3[i]
        camarilla_l3_val = camarilla_l3[i]
        
        # Trend filter: price relative to 4h EMA20
        uptrend = curr_close > ema_20_val
        downtrend = curr_close < ema_20_val
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        volume_confirm = curr_volume > 1.5 * vol_ma
        
        if position == 0:
            # Look for breakout signals
            # Long: price breaks above Camarilla H3 with volume confirmation in uptrend
            long_breakout = (curr_close > camarilla_h3_val) and volume_confirm and uptrend
            # Short: price breaks below Camarilla L3 with volume confirmation in downtrend
            short_breakout = (curr_close < camarilla_l3_val) and volume_confirm and downtrend
            
            if long_breakout:
                signals[i] = 0.20
                position = 1
                entry_price = curr_close
            elif short_breakout:
                signals[i] = -0.20
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Stoploss: 2 * ATR below entry
            stop_loss = entry_price - 2.0 * atr_val
            # Exit conditions: price closes below Camarilla L3 OR stoploss hit OR EMA20 trend turns down
            if curr_close < camarilla_l3_val or curr_close < stop_loss or curr_close < ema_20_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short position management
            # Stoploss: 2 * ATR above entry
            stop_loss = entry_price + 2.0 * atr_val
            # Exit conditions: price closes above Camarilla H3 OR stoploss hit OR EMA20 trend turns up
            if curr_close > camarilla_h3_val or curr_close > stop_loss or curr_close > ema_20_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_H3L3_Breakout_4hEMA20_Trend_VolumeSpike_Session"
timeframe = "1h"
leverage = 1.0