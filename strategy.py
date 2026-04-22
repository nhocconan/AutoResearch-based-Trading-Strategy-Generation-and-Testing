#!/usr/bin/env python3

"""
Hypothesis: 12-hour Camarilla Pivot Breakout with 1-day trend filter and volume confirmation.
Camarilla pivots provide precise support/resistance levels based on previous day's action.
Breakouts above R3 or below S3 indicate strong momentum with institutional backing.
1-day EMA filter ensures alignment with daily trend to avoid counter-trend trades.
Volume spikes confirm participation at breakout points.
Designed for low-frequency, high-conviction trades in both bull and bear markets.
Target: 12-37 trades/year per symbol (50-150 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data for Camarilla calculation - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 12h bar
    # H, L, C from previous completed 12h bar
    prev_high = df_12h['high'].shift(1).values  # Previous bar's high
    prev_low = df_12h['low'].shift(1).values    # Previous bar's low
    prev_close = df_12h['close'].shift(1).values # Previous bar's close
    
    # Camarilla formulas
    # R4 = C + ((H-L) * 1.1/2)
    # R3 = C + ((H-L) * 1.1/4)
    # R2 = C + ((H-L) * 1.1/6)
    # R1 = C + ((H-L) * 1.1/12)
    # S1 = C - ((H-L) * 1.1/12)
    # S2 = C - ((H-L) * 1.1/6)
    # S3 = C - ((H-L) * 1.1/4)
    # S4 = C - ((H-L) * 1.1/2)
    
    range_hl = prev_high - prev_low
    r3 = prev_close + (range_hl * 1.1 / 4)
    s3 = prev_close - (range_hl * 1.1 / 4)
    r4 = prev_close + (range_hl * 1.1 / 2)
    s4 = prev_close - (range_hl * 1.1 / 2)
    
    # Align Camarilla levels to 12h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_12h, r3)
    s3_aligned = align_htf_to_ltf(prices, df_12h, s3)
    r4_aligned = align_htf_to_ltf(prices, df_12h, r4)
    s4_aligned = align_htf_to_ltf(prices, df_12h, s4)
    
    # Load 1d data for trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d EMA for trend filter (34-period)
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 12h volume average (24-period = 12 days)
    vol_avg_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(24, n):  # Start after enough data for volume average
        # Skip if data not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_avg_24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R3, above 1d EMA, volume spike
            if (close[i] > r3_aligned[i] and                    # Price above R3
                close[i] > ema_34_1d_aligned[i] and         # Above 1d EMA (bullish trend)
                volume[i] > 2.0 * vol_avg_24[i]):           # Volume spike
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3, below 1d EMA, volume spike
            elif (close[i] < s3_aligned[i] and               # Price below S3
                  close[i] < ema_34_1d_aligned[i] and       # Below 1d EMA (bearish trend)
                  volume[i] > 2.0 * vol_avg_24[i]):         # Volume spike
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price returns to opposite S/R level or crosses 1d EMA
            exit_signal = False
            
            if position == 1:
                # Exit long: price drops below S3 or below 1d EMA
                if close[i] < s3_aligned[i] or close[i] < ema_34_1d_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price rises above R3 or above 1d EMA
                if close[i] > r3_aligned[i] or close[i] > ema_34_1d_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_Camarilla_R3S3_Breakout_1dEMA34_Volume"
timeframe = "12h"
leverage = 1.0