#!/usr/bin/env python3
"""
1h_Camarilla_R1S1_Breakout_4hTrend_1dVolume
Hypothesis: Combines Camarilla pivot points (R1/S1) on 1h for entry timing with 4h trend filter (EMA50) and 1d volume confirmation to capture breakouts with institutional interest. The Camarilla levels provide precise entry points near intraday support/resistance, while higher timeframe filters ensure alignment with dominant trend and reduce false breakouts. Designed for low frequency (15-35 trades/year) to minimize fee drag in 1h timeframe.
"""
name = "1h_Camarilla_R1S1_Breakout_4hTrend_1dVolume"
timeframe = "1h"
leverage = 1.0

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
    
    # Calculate Camarilla pivot points for 1h
    # Using previous period's high, low, close
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    pivot = (prev_high + prev_low + prev_close) / 3
    range_val = prev_high - prev_low
    
    # Camarilla levels: R1 = close + (high-low)*1.1/12, S1 = close - (high-low)*1.1/12
    r1 = prev_close + range_val * 1.1 / 12
    s1 = prev_close - range_val * 1.1 / 12
    
    # 4h EMA50 for trend filter
    df_4h = get_htf_data(prices, '4h')
    ema_50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # 1d volume filter: current volume > 1.3 * 20-period average
    df_1d = get_htf_data(prices, '1d')
    vol_avg_1d = pd.Series(df_1d['volume']).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    volume_filter = volume > (vol_avg_1d_aligned * 1.3)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need enough data for calculations
    
    for i in range(start_idx, n):
        # Skip if any data is not ready or outside session
        if (np.isnan(r1[i]) or np.isnan(s1[i]) or np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(vol_avg_1d_aligned[i]) or not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R1 + 4h uptrend + volume confirmation
            if close[i] > r1[i] and close[i] > ema_50_4h_aligned[i] and volume_filter[i]:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below S1 + 4h downtrend + volume confirmation
            elif close[i] < s1[i] and close[i] < ema_50_4h_aligned[i] and volume_filter[i]:
                signals[i] = -0.20
                position = -1
        elif position != 0:
            # Exit: price returns to opposite Camarilla level (S1 for long, R1 for short)
            if position == 1:
                if close[i] <= s1[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
            else:  # position == -1
                if close[i] >= r1[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
    
    return signals