#!/usr/bin/env python3
"""
12h_Camarilla_R1S1_Breakout_1wTrend_VolumeRegime
Hypothesis: 12-hour Camarilla R1/S1 breakout with 1-week trend filter (price > 1w EMA50) and volume regime filter (>1.5x 20-period average volume).
Long when price breaks above R1 in 1-week uptrend with volume confirmation.
Short when price breaks below S1 in 1-week downtrend with volume confirmation.
Exit via ATR trailing stop (2.5*ATR from extreme) or opposite Camarilla level (S1 for longs, R1 for shorts).
Camarilla levels provide precise intraday support/resistance derived from prior day's range.
Weekly trend filter ensures alignment with higher timeframe bias, reducing counter-trend trades.
Volume regime confirms breakouts have conviction. Designed for ~75-125 trades over 4 years (19-31/year) via tight breakout conditions.
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
    
    # Get 1d data for Camarilla calculation (prior day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:  # need 50 for EMA50
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA50 for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate ATR for stoploss (14-period)
    atr_period = 14
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    # Volume regime: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_regime = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    long_extreme = 0.0   # highest close since long entry
    short_extreme = 0.0  # lowest close since short entry
    
    # Start index: need warmup for calculations
    start_idx = max(100, atr_period, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        ema_trend = ema_50_1w_aligned[i]
        
        # Calculate Camarilla levels using prior 1d bar (i-1 in 1d timeframe)
        # We need the 1d bar that completed before current 12h bar
        # Since we're on 12h timeframe, we look back 2 12h bars to get prior day
        # But safer: use the 1d data directly with proper alignment
        if i >= 2:  # need at least 2 bars to have prior completed 1d bar
            # Get the 1d index that corresponds to the prior completed day
            # We'll use the 1d bar that ended before the current 12h bar started
            # Since 12h bars are half of 1d bars, we can use integer division
            idx_1d = i // 2
            if idx_1d > 0 and idx_1d < len(df_1d):
                # Prior completed 1d bar
                high_1d = df_1d['high'].values[idx_1d - 1]
                low_1d = df_1d['low'].values[idx_1d - 1]
                close_1d = df_1d['close'].values[idx_1d - 1]
                
                # Calculate Camarilla levels
                range_1d = high_1d - low_1d
                if range_1d > 0:
                    r1 = close_1d + (range_1d * 1.1 / 12)
                    s1 = close_1d - (range_1d * 1.1 / 12)
                    r3 = close_1d + (range_1d * 1.1 / 4)
                    s3 = close_1d - (range_1d * 1.1 / 4)
                else:
                    r1 = s1 = r3 = s3 = close_1d
            else:
                # Not enough 1d data yet
                signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
                continue
        else:
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:
            # Only trade in trending regimes (1w EMA50 filter)
            if close[i] > ema_trend:  # 1w uptrend regime
                # Long: break above R1 with volume confirmation
                long_signal = (close[i] > r1) and vol_regime[i]
            else:  # 1w downtrend regime
                # Short: break below S1 with volume confirmation
                short_signal = (close[i] < s1) and vol_regime[i]
            
            if 'long_signal' in locals() and long_signal:
                signals[i] = 0.25
                position = 1
                long_extreme = close[i]
            elif 'short_signal' in locals() and short_signal:
                signals[i] = -0.25
                position = -1
                short_extreme = close[i]
            else:
                signals[i] = 0.0
                # Clear signal variables for next iteration
                if 'long_signal' in locals(): del long_signal
                if 'short_signal' in locals(): del short_signal
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Update extreme for trailing stop
            if close[i] > long_extreme:
                long_extreme = close[i]
            # Exit conditions: 
            # 1. ATR trailing stop (2.5*ATR from extreme)
            atr_stop = long_extreme - 2.5 * atr[i]
            # 2. Price breaks below S1 (opposite Camarilla level)
            if close[i] <= atr_stop or close[i] < s1:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Update extreme for trailing stop
            if close[i] < short_extreme:
                short_extreme = close[i]
            # Exit conditions:
            # 1. ATR trailing stop (2.5*ATR from extreme)
            atr_stop = short_extreme + 2.5 * atr[i]
            # 2. Price breaks above R1 (opposite Camarilla level)
            if close[i] >= atr_stop or close[i] > r1:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R1S1_Breakout_1wTrend_VolumeRegime"
timeframe = "12h"
leverage = 1.0