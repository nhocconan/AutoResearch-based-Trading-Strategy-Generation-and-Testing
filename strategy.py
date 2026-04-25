#!/usr/bin/env python3
"""
12h_Camarilla_R3S3_Breakout_1dTrend_Filter
Hypothesis: 12h Camarilla R3/S3 breakouts with 1d EMA34 trend filter. 
Only trade breakouts in direction of daily trend to avoid counter-trend whipsaws.
Uses volume confirmation (>1.5x 20-period average) to filter low-quality breakouts.
Designed for low trade frequency (~15-25/year) to work in both bull and bear markets via trend alignment.
Camarilla levels provide precise intraday support/resistance with high breakout fidelity.
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
    
    # Get 1d data for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate EMA34 on 1d close for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d, additional_delay_bars=1)
    
    # Calculate 20-period volume average for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA34 (34) and volume MA (20)
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(ema34_aligned[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Calculate Camarilla levels for 12h timeframe using previous bar's OHLC
        if i >= 1:
            prev_high = high[i-1]
            prev_low = low[i-1]
            prev_close = close[i-1]
            daily_range = prev_high - prev_low
            
            if daily_range > 0:  # Avoid division by zero
                camarilla_r3 = prev_close + daily_range * 1.1 / 4
                camarilla_s3 = prev_close - daily_range * 1.1 / 4
            else:
                camarilla_r3 = prev_close
                camarilla_s3 = prev_close
        else:
            camarilla_r3 = close[i]
            camarilla_s3 = close[i]
        
        if position == 0:
            # Look for Camarilla breakout signals with trend and volume filters
            # Long: price breaks above R3 in uptrend (close > EMA34) with volume confirmation
            # Short: price breaks below S3 in downtrend (close < EMA34) with volume confirmation
            volume_ok = volume[i] > 1.5 * vol_ma[i]
            long_signal = (close[i] > camarilla_r3) and (close[i] > ema34_aligned[i]) and volume_ok
            short_signal = (close[i] < camarilla_s3) and (close[i] < ema34_aligned[i]) and volume_ok
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit when price moves back below EMA34 (trend reversal)
            exit_signal = close[i] < ema34_aligned[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when price moves back above EMA34 (trend reversal)
            exit_signal = close[i] > ema34_aligned[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R3S3_Breakout_1dTrend_Filter"
timeframe = "12h"
leverage = 1.0