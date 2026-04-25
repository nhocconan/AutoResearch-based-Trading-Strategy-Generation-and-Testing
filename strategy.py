#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike
Hypothesis: Camarilla R1/S1 breakout on 4h with 1d EMA50 trend filter and volume spike confirmation.
Long when price breaks above R1 in uptrend (close > daily EMA50) with volume spike.
Short when price breaks below S1 in downtrend (close < daily EMA50) with volume spike.
Exit when price re-enters Camarilla H3/L3 range or trend reverses.
Designed for low trade frequency (~20-50/year) and robustness in both bull and bear markets.
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
    
    # Get 4h data for Camarilla calculations (primary timeframe)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Camarilla levels for each 4h bar using previous bar's OHLC
    # Camarilla levels: based on previous day's range
    prev_close = np.roll(close_4h, 1)
    prev_high = np.roll(high_4h, 1)
    prev_low = np.roll(low_4h, 1)
    prev_close[0] = close_4h[0]  # first bar uses current close as previous
    prev_high[0] = high_4h[0]
    prev_low[0] = low_4h[0]
    
    range_4h = prev_high - prev_low
    camarilla_h3 = prev_close + range_4h * 1.1 / 4
    camarilla_l3 = prev_close - range_4h * 1.1 / 4
    camarilla_h4 = prev_close + range_4h * 1.1 / 2
    camarilla_l4 = prev_close - range_4h * 1.1 / 2
    camarilla_h5 = prev_close + range_4h * 1.1
    camarilla_l5 = prev_close - range_4h * 1.1
    camarilla_r1 = prev_close + range_4h * 1.1 / 12
    camarilla_s1 = prev_close - range_4h * 1.1 / 12
    camarilla_r2 = prev_close + range_4h * 1.1 / 6
    camarilla_s2 = prev_close - range_4h * 1.1 / 6
    camarilla_r3 = prev_close + range_4h * 1.1 / 4
    camarilla_s3 = prev_close - range_4h * 1.1 / 4
    
    # Align Camarilla levels to original timeframe
    h3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_h3)
    l3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_l3)
    h4_aligned = align_htf_to_ltf(prices, df_4h, camarilla_h4)
    l4_aligned = align_htf_to_ltf(prices, df_4h, camarilla_l4)
    r1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r1)
    s1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s1)
    
    # Get daily data for trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Daily EMA50 for trend direction
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for calculations
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        ema_trend = ema_50_1d_aligned[i]
        
        if position == 0:
            # Regime-based entry logic
            if close[i] > ema_trend:  # Uptrend regime (daily)
                # Long: break above R1 with volume spike
                long_signal = (close[i] > r1_aligned[i]) and vol_spike[i]
                # Short: break below S1 only if extreme volume spike (counter-trend fade)
                short_signal = (close[i] < s1_aligned[i]) and vol_spike[i] and (volume[i] > (4.0 * vol_ma_20[i]))
            else:  # Downtrend regime (daily)
                # Short: break below S1 with volume spike
                short_signal = (close[i] < s1_aligned[i]) and vol_spike[i]
                # Long: break above R1 only if extreme volume spike (counter-trend fade)
                long_signal = (close[i] > r1_aligned[i]) and vol_spike[i] and (volume[i] > (4.0 * vol_ma_20[i]))
            
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
            # Exit conditions: re-enter H3/L3 range or trend reversal
            exit_signal = (close[i] < h3_aligned[i]) or (close[i] < ema_trend * 0.99)
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit conditions: re-enter H3/L3 range or trend reversal
            exit_signal = (close[i] > l3_aligned[i]) or (close[i] > ema_trend * 1.01)
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0