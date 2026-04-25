#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_12hEMA50_Trend_VolumeS
Hypothesis: Camarilla R1/S1 breakout on 4h with 12h EMA50 trend filter and volume confirmation.
Only trade breakouts in direction of 12h trend with volume > 1.5x average. Uses discrete position sizing (0.30) 
to minimize fee churn. Designed for moderate trade frequency (~30-50/year) to work in both bull and bear markets 
via trend alignment and volume confirmation. Camarilla levels provide high-probability reversal/breakout points.
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
    
    # Get 12h data for HTF trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate EMA50 on 12h close for trend filter
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 1d data for Camarilla levels (yesterday's OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate average volume for confirmation (20-period SMA)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 4h timeframe
    ema50_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h, additional_delay_bars=1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA50 (50) and volume SMA (20)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_aligned[i]) or 
            np.isnan(vol_sma_20[i]) or
            i < 1):  # need previous day for Camarilla
            signals[i] = 0.0 if position == 0 else (0.30 if position == 1 else -0.30)
            continue
        
        # Get previous day's OHLC for Camarilla calculation
        prev_day_idx = i - 1
        if prev_day_idx < 0:
            signals[i] = 0.0 if position == 0 else (0.30 if position == 1 else -0.30)
            continue
            
        # Calculate Camarilla levels from previous day's OHLC
        # We need to get the actual previous day's values from 1d data
        # Find the index in df_1d that corresponds to the previous day
        prev_open_1d = None
        prev_high_1d = None
        prev_low_1d = None
        prev_close_1d = None
        
        # Simple approach: use the most recent completed 1d candle
        # Since we're on 4h timeframe, we can approximate by looking back
        # For simplicity, we'll use a rolling window on the 4h data to get daily OHLC
        # This is not perfect but avoids look-ahead and uses available data
        
        # Instead, let's calculate Camarilla levels using a simpler approach:
        # Use the previous 6x4h candles to approximate daily OHLC (6*4h = 24h)
        lookback = min(6, i)  # max 6 periods back (24h)
        if lookback < 6 and i < 6:
            signals[i] = 0.0 if position == 0 else (0.30 if position == 1 else -0.30)
            continue
            
        start_lookback = i - lookback
        day_high = np.max(high[start_lookback:i])
        day_low = np.min(low[start_lookback:i])
        day_close = close[i-1]  # previous close
        day_open = open_prices[start_lookback] if 'open_prices' in locals() else close[start_lookback]
        
        # We need open prices - let's extract them
        if 'open_prices' not in locals():
            open_prices = prices['open'].values
            
        day_open = open_prices[start_lookback]
        
        # Calculate Camarilla levels
        range_val = day_high - day_low
        if range_val <= 0:
            signals[i] = 0.0 if position == 0 else (0.30 if position == 1 else -0.30)
            continue
            
        camarilla_r1 = day_close + (range_val * 1.1 / 12)
        camarilla_s1 = day_close - (range_val * 1.1 / 12)
        camarilla_r2 = day_close + (range_val * 1.1 / 6)
        camarilla_s2 = day_close - (range_val * 1.1 / 6)
        camarilla_r3 = day_close + (range_val * 1.1 / 4)
        camarilla_s3 = day_close - (range_val * 1.1 / 4)
        camarilla_r4 = day_close + (range_val * 1.1 / 2)
        camarilla_s4 = day_close - (range_val * 1.1 / 2)
        
        # Volume confirmation: current volume > 1.5x average
        volume_confirm = volume[i] > (1.5 * vol_sma_20[i])
        
        if position == 0:
            # Look for breakout signals with trend and volume confirmation
            # Long: price breaks above R1 in uptrend (close > EMA50) with volume
            # Short: price breaks below S1 in downtrend (close < EMA50) with volume
            long_signal = (close[i] > camarilla_r1) and (close[i] > ema50_aligned[i]) and volume_confirm
            short_signal = (close[i] < camarilla_s1) and (close[i] < ema50_aligned[i]) and volume_confirm
            
            if long_signal:
                signals[i] = 0.30
                position = 1
            elif short_signal:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.30
            # Exit when price moves back below R1 (failed breakout) or trend reverses
            exit_signal = (close[i] < camarilla_r1) or (close[i] < ema50_aligned[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.30
            # Exit when price moves back above S1 (failed breakout) or trend reverses
            exit_signal = (close[i] > camarilla_s1) or (close[i] > ema50_aligned[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_12hEMA50_Trend_VolumeS"
timeframe = "4h"
leverage = 1.0