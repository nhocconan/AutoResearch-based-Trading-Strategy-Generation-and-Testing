#!/usr/bin/env python3
"""
1h_Camarilla_R1S1_Breakout_4hTrend_VolumeSpike
Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA50 trend filter and volume spike confirmation.
Long when price breaks above R1 in uptrend (close > 4h EMA50) with volume spike (>2.0x 20-bar average).
Short when price breaks below S1 in downtrend (close < 4h EMA50) with volume spike.
Exit when price re-enters H3-L3 range or trend reverses.
Uses 4h for signal direction, 1h only for entry timing to minimize fee drag.
Designed for 15-37 trades/year on 1h timeframe with tight entry conditions.
Works in bull markets via trend-following breakouts and in bear markets via counter-trend fades on extreme volume spikes.
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
    
    # Get 4h data for Camarilla pivot calculation and trend filter (HTF)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Camarilla levels for previous 4h bar
    prev_close = np.concatenate([[np.nan], close_4h[:-1]])
    prev_high = np.concatenate([[np.nan], high_4h[:-1]])
    prev_low = np.concatenate([[np.nan], low_4h[:-1]])
    
    pivot = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    r1 = prev_close + range_hl * 1.1 / 12
    s1 = prev_close - range_hl * 1.1 / 12
    h3 = prev_close + range_hl * 1.1 / 4
    l3 = prev_close - range_hl * 1.1 / 4
    
    # Align Camarilla levels to 1h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_4h, r1)
    s1_aligned = align_htf_to_ltf(prices, df_4h, s1)
    h3_aligned = align_htf_to_ltf(prices, df_4h, h3)
    l3_aligned = align_htf_to_ltf(prices, df_4h, l3)
    
    # Get 4h data for trend filter (EMA50)
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Volume confirmation: volume > 2.0x 20-period average (1h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_ma_20)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for calculations
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if not in trading session or data not ready
        if not in_session[i] or (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
                                 np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or 
                                 np.isnan(ema_50_4h_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        ema_trend = ema_50_4h_aligned[i]
        
        if position == 0:
            # Regime-based entry logic
            if close[i] > ema_trend:  # Uptrend regime (4h)
                # Long: break above R1 with volume spike
                long_signal = (close[i] > r1_aligned[i]) and vol_spike[i]
                # Short: break below S1 only if extreme volume spike (counter-trend fade)
                short_signal = (close[i] < s1_aligned[i]) and vol_spike[i] and (volume[i] > (4.0 * vol_ma_20[i]))
            else:  # Downtrend regime (4h)
                # Short: break below S1 with volume spike
                short_signal = (close[i] < s1_aligned[i]) and vol_spike[i]
                # Long: break above R1 only if extreme volume spike (counter-trend fade)
                long_signal = (close[i] > r1_aligned[i]) and vol_spike[i] and (volume[i] > (4.0 * vol_ma_20[i]))
            
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
            # Exit conditions: re-enter H3-L3 range or trend reversal
            exit_signal = (close[i] < h3_aligned[i] and close[i] > l3_aligned[i]) or (close[i] < ema_trend * 0.99)
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.20
            # Exit conditions: re-enter H3-L3 range or trend reversal
            exit_signal = (close[i] > l3_aligned[i] and close[i] < h3_aligned[i]) or (close[i] > ema_trend * 1.01)
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_Camarilla_R1S1_Breakout_4hTrend_VolumeSpike"
timeframe = "1h"
leverage = 1.0