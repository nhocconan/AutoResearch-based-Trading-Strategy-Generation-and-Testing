#!/usr/bin/env python3
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
    
    # Load 1-day data for pivot and ATR
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily range
    daily_range = high_1d - low_1d
    
    # Calculate Camarilla R1 and S1 from previous day (avoid look-ahead)
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_range = np.roll(daily_range, 1)
    prev_close[0] = np.nan
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_range[0] = np.nan
    
    r1 = prev_close + (prev_range * 1.1 / 12)
    s1 = prev_close - (prev_range * 1.1 / 12)
    
    # Calculate 1-day ATR(14) for volatility filter and stop
    tr1 = np.maximum(high_1d - low_1d,
                     np.maximum(np.abs(high_1d - np.roll(close_1d, 1)),
                                np.abs(low_1d - np.roll(close_1d, 1))))
    tr1[0] = np.nan
    atr14_1d = pd.Series(tr1).rolling(window=14, min_periods=14).mean().values
    
    # Volume spike filter (20-period on 12h)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma20
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Align indicators to 12-hour timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    atr14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr14_1d)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(100, n):
        # Skip if data not ready or outside session
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(atr14_1d_aligned[i]) or np.isnan(vol_ma20[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above R1 + volume spike
            if close[i] > r1_aligned[i] and vol_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S1 + volume spike
            elif close[i] < s1_aligned[i] and vol_spike[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price returns to opposite level or ATR-based stop
            if position == 1:
                # Exit long if price crosses below S1 or ATR stop hit
                if close[i] < s1_aligned[i] or close[i] < (high[max(0, i-1)] - 1.5 * atr14_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short if price crosses above R1 or ATR stop hit
                if close[i] > r1_aligned[i] or close[i] > (low[max(0, i-1)] + 1.5 * atr14_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_Volume_ATRStop"
timeframe = "12h"
leverage = 1.0