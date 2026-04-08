#!/usr/bin/env python3
# 12h_atr_breakout_1d_trend_volume_v1
# Hypothesis: On 12h timeframe, use ATR breakout from previous candle's range with 1d trend filter and volume confirmation.
# Long when price breaks above (previous close + 0.5 * ATR) with volume > 1.3x average and 1d uptrend.
# Short when price breaks below (previous close - 0.5 * ATR) with volume > 1.3x average and 1d downtrend.
# Exit when price crosses previous close (mean reversion) or volume drops below average.
# This strategy targets 15-35 trades/year by using 12h timeframe and strict breakout conditions.
# Works in both bull and bear markets via trend filter and volatility-based entry.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_atr_breakout_1d_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate ATR (14-period) on 12h
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0  # First period has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d trend filter: EMA20
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 20:
        return np.zeros(n)
    
    daily_close = df_daily['close'].values
    daily_ema20 = pd.Series(daily_close).ewm(span=20, min_periods=20, adjust=False).mean().values
    daily_ema20_12h = align_htf_to_ltf(prices, df_daily, daily_ema20)
    
    # Volume confirmation: 20-period average on 12h
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if data not available
        if np.isnan(atr[i]) or np.isnan(daily_ema20_12h[i]) or np.isnan(avg_volume[i]):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below previous close (mean reversion) or volume drops below average
            if close[i] <= close[i-1] or volume[i] < avg_volume[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above previous close (mean reversion) or volume drops below average
            if close[i] >= close[i-1] or volume[i] < avg_volume[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Calculate breakout levels: previous close +/- 0.5 * ATR
            upper_break = close[i-1] + 0.5 * atr[i]
            lower_break = close[i-1] - 0.5 * atr[i]
            
            # Volume confirmation: current volume > 1.3x average volume
            volume_ok = volume[i] > 1.3 * avg_volume[i]
            
            # Daily trend filter
            daily_uptrend = close[i] > daily_ema20_12h[i]
            daily_downtrend = close[i] < daily_ema20_12h[i]
            
            # Long entry: price breaks above upper level with volume and uptrend
            if close[i] > upper_break and volume_ok and daily_uptrend:
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below lower level with volume and downtrend
            elif close[i] < lower_break and volume_ok and daily_downtrend:
                position = -1
                signals[i] = -0.25
    
    return signals