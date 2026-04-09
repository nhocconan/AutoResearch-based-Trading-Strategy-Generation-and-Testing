#!/usr/bin/env python3
# 1h_htf_trend_timing_v1
# Hypothesis: Use 4h/1d trend alignment for signal direction, 1h for precise entry timing with volume confirmation and session filter.
# Long when: 4h EMA21 > 4h EMA50 AND 1d close > 1d EMA50 AND 1h close > 1h EMA20 AND volume > 1.5x 20-period average AND hour 08-20 UTC.
# Short when: 4h EMA21 < 4h EMA50 AND 1d close < 1d EMA50 AND 1h close < 1h EMA20 AND volume > 1.5x 20-period average AND hour 08-20 UTC.
# Exit when trend alignment breaks or reverse signal occurs.
# Uses discrete position sizing (0.20) to minimize fee churn.
# Target: 15-37 trades/year (60-150 total over 4 years) on BTC/ETH/SOL.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_htf_trend_timing_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # 1h EMA20 for entry timing
    close_s = pd.Series(close)
    ema20_1h = close_s.ewm(span=20, min_periods=20, adjust=False).mean().values
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    close_s_4h = pd.Series(close_4h)
    ema21_4h = close_s_4h.ewm(span=21, min_periods=21, adjust=False).mean().values
    ema50_4h = close_s_4h.ewm(span=50, min_periods=50, adjust=False).mean().values
    ema21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema21_4h)
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    close_s_1d = pd.Series(close_1d)
    ema50_1d = close_s_1d.ewm(span=50, min_periods=50, adjust=False).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(ema21_4h_aligned[i]) or np.isnan(ema50_4h_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(ema20_1h[i]) or
            np.isnan(volume_ma[i]) or np.isnan(close[i]) or np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        # Session filter
        if not in_session[i]:
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        if position == 1:  # Long position
            # Exit: 4h/1d trend breaks down OR reverse signal
            trend_broken = (ema21_4h_aligned[i] <= ema50_4h_aligned[i]) or (close[i] <= ema20_1h[i])
            reverse_signal = (ema21_4h_aligned[i] < ema50_4h_aligned[i]) and (close_1d_aligned_val := close_s_1d.iloc[-1] if hasattr(close_s_1d, 'iloc') else close_s_1d[-1]) < ema50_1d_aligned[i] and (close[i] < ema20_1h[i]) and volume_confirmed
            if trend_broken:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: 4h/1d trend breaks up OR reverse signal
            trend_broken = (ema21_4h_aligned[i] >= ema50_4h_aligned[i]) or (close[i] >= ema20_1h[i])
            reverse_signal = (ema21_4h_aligned[i] > ema50_4h_aligned[i]) and (close_1d_aligned_val := close_s_1d.iloc[-1] if hasattr(close_s_1d, 'iloc') else close_s_1d[-1]) > ema50_1d_aligned[i] and (close[i] > ema20_1h[i]) and volume_confirmed
            if trend_broken:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Check for trend alignment and entry conditions
            long_trend = (ema21_4h_aligned[i] > ema50_4h_aligned[i]) and (close_s_1d.iloc[-1] if hasattr(close_s_1d, 'iloc') else close_s_1d[-1]) > ema50_1d_aligned[i]
            short_trend = (ema21_4h_aligned[i] < ema50_4h_aligned[i]) and (close_s_1d.iloc[-1] if hasattr(close_s_1d, 'iloc') else close_s_1d[-1]) < ema50_1d_aligned[i]
            
            # Get current 1d EMA50 aligned value
            ema50_1d_current = ema50_1d_aligned[i]
            
            long_entry = long_trend and (close[i] > ema20_1h[i]) and volume_confirmed
            short_entry = short_trend and (close[i] < ema20_1h[i]) and volume_confirmed
            
            if long_entry:
                position = 1
                signals[i] = 0.20
            elif short_entry:
                position = -1
                signals[i] = -0.20
    
    return signals