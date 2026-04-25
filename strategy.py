#!/usr/bin/env python3
"""
1h_4h_1d_Camarilla_R1S1_Breakout_Trend_Volume
Hypothesis: Camarilla R1/S1 breakout on 1h with 4h trend filter and 1d volume spike.
Long when price breaks above R1 in uptrend (4h close > 4h EMA50) with 1d volume > 2x 20-day average.
Short when price breaks below S1 in downtrend (4h close < 4h EMA50) with volume spike.
Exit when price re-enters Camarilla H3/L3 range or trend reverses.
Designed for 1h timeframe: use 4h for signal direction, 1d for volume regime, 1h only for entry timing.
Target: 60-150 total trades over 4 years = 15-37/year for 1h.
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
    
    # Get 4h data for trend filter and Camarilla calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Camarilla levels (based on previous 4h bar)
    # Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # H3 = C + (H-L)*1.1/4, L3 = C - (H-L)*1.1/4
    prev_close_4h = np.roll(close_4h, 1)
    prev_high_4h = np.roll(high_4h, 1)
    prev_low_4h = np.roll(low_4h, 1)
    prev_close_4h[0] = close_4h[0]  # first bar
    prev_high_4h[0] = high_4h[0]
    prev_low_4h[0] = low_4h[0]
    
    range_4h = prev_high_4h - prev_low_4h
    R1 = prev_close_4h + (range_4h * 1.1 / 12)
    S1 = prev_close_4h - (range_4h * 1.1 / 12)
    H3 = prev_close_4h + (range_4h * 1.1 / 4)
    L3 = prev_close_4h - (range_4h * 1.1 / 4)
    
    # Align Camarilla levels to 1h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_4h, R1)
    S1_aligned = align_htf_to_ltf(prices, df_4h, S1)
    H3_aligned = align_htf_to_ltf(prices, df_4h, H3)
    L3_aligned = align_htf_to_ltf(prices, df_4h, L3)
    
    # 4h EMA50 for trend filter
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Get 1d data for volume spike filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = volume_1d > (2.0 * vol_ma_20_1d)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    
    # Session filter: 08-20 UTC (reduce noise trades)
    hours = prices.index.hour  # prices.index is DatetimeIndex
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for calculations
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
            
        # Skip if data not ready
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or 
            np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or
            np.isnan(ema_50_4h_aligned[i]) or np.isnan(vol_spike_1d_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        ema_trend = ema_50_4h_aligned[i]
        vol_spike = vol_spike_1d_aligned[i]
        
        if position == 0:
            # Regime-based entry logic
            if close[i] > ema_trend:  # Uptrend regime (4h)
                # Long: break above R1 with volume spike
                long_signal = (close[i] > R1_aligned[i]) and vol_spike
                # Short: break below S1 only if extreme volume spike (counter-trend fade)
                short_signal = (close[i] < S1_aligned[i]) and vol_spike and (volume[i] > (4.0 * np.mean(volume[max(0,i-20):i+1])))
            else:  # Downtrend regime (4h)
                # Short: break below S1 with volume spike
                short_signal = (close[i] < S1_aligned[i]) and vol_spike
                # Long: break above R1 only if extreme volume spike (counter-trend fade)
                long_signal = (close[i] > R1_aligned[i]) and vol_spike and (volume[i] > (4.0 * np.mean(volume[max(0,i-20):i+1])))
            
            if long_signal:
                signals[i] = 0.20
                position = 1
            elif short_signal:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.20
            # Exit conditions: re-enter H3/L3 range or trend reversal
            exit_signal = (close[i] < H3_aligned[i]) or (close[i] < ema_trend * 0.995)
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.20
            # Exit conditions: re-enter H3/L3 range or trend reversal
            exit_signal = (close[i] > L3_aligned[i]) or (close[i] > ema_trend * 1.005)
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_4h_1d_Camarilla_R1S1_Breakout_Trend_Volume"
timeframe = "1h"
leverage = 1.0