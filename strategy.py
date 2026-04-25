#!/usr/bin/env python3
"""
6h_Camarilla_R3S3_Breakout_12hTrend_VolumeConfirm
Hypothesis: 6h Camarilla R3/S3 breakout with 12h EMA50 trend filter and volume confirmation (>1.8x 20-bar avg). Enters long when price breaks above R3 in 12h uptrend, short when breaks below S3 in 12h downtrend. Uses ATR-based trailing stop (2.5x ATR) and discrete sizing (0.25) to limit fee churn. Designed for 6h timeframe with ~12-30 trades/year, works in bull/bear by following 12h trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 12h data for HTF trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # 12h EMA50 for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate ATR for stoploss (using 14 periods)
    tr1 = pd.Series(high - low).values
    tr2 = pd.Series(np.abs(high - np.roll(close, 1))).values
    tr3 = pd.Series(np.abs(low - np.roll(close, 1))).values
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume spike: current volume > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    atr_stop = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Start index: need 20-period data for volume MA and 50 for 12h EMA
    start_idx = max(20, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        
        # Calculate Camarilla levels for current 6h bar using previous 12h bar's OHLC
        # Get the previous completed 12h bar index
        if i >= 1:
            prev_close_12h = close_12h[i-1] if (i-1) < len(close_12h) else close_12h[-1]
            prev_high_12h = high_12h[i-1] if (i-1) < len(high_12h) else high_12h[-1]
            prev_low_12h = low_12h[i-1] if (i-1) < len(low_12h) else low_12h[-1]
        else:
            prev_close_12h = close_12h[0]
            prev_high_12h = high_12h[0]
            prev_low_12h = low_12h[0]
        
        # Camarilla levels calculation (R3/S3)
        range_12h = prev_high_12h - prev_low_12h
        camarilla_r3 = prev_close_12h + (range_12h * 1.1 / 4)
        camarilla_s3 = prev_close_12h - (range_12h * 1.1 / 4)
        
        if position == 0:
            # Long: price breaks above R3 in 12h uptrend with volume confirmation
            bullish_breakout = (curr_close > camarilla_r3) and \
                              (close_12h[i] > ema_50_12h_aligned[i]) and \
                              volume_spike[i]
            # Short: price breaks below S3 in 12h downtrend with volume confirmation
            bearish_breakout = (curr_close < camarilla_s3) and \
                              (close_12h[i] < ema_50_12h_aligned[i]) and \
                              volume_spike[i]
            
            if bullish_breakout:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                atr_stop = entry_price - (2.5 * atr[i])
                highest_since_entry = curr_close
            elif bearish_breakout:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
                atr_stop = entry_price + (2.5 * atr[i])
                lowest_since_entry = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position with trailing stop
            signals[i] = 0.25
            highest_since_entry = max(highest_since_entry, curr_close)
            # Exit: price breaks below S3 OR trailing stop hit OR trend turns down
            trailing_stop = highest_since_entry - (2.5 * atr[i])
            if (curr_close < camarilla_s3) or \
               (curr_close < trailing_stop) or \
               (close_12h[i] < ema_50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position with trailing stop
            signals[i] = -0.25
            lowest_since_entry = min(lowest_since_entry, curr_close)
            # Exit: price breaks above R3 OR trailing stop hit OR trend turns up
            trailing_stop = lowest_since_entry + (2.5 * atr[i])
            if (curr_close > camarilla_r3) or \
               (curr_close > trailing_stop) or \
               (close_12h[i] > ema_50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Camarilla_R3S3_Breakout_12hTrend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0