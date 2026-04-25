#!/usr/bin/env python3
"""
1h_Camarilla_H3L3_Breakout_4hTrend_1dVolSpike
Hypothesis: Trade 1h timeframe using Camarilla H3/L3 breakout for entry timing, 4h EMA50 for trend filter, 
and daily volume spike (>2.0x 20-bar MA) for confirmation. Enter long when price breaks above H3 AND 
above 4h EMA50 AND volume spike. Enter short when price breaks below L3 AND below 4h EMA50 AND volume spike. 
Exit on opposite Camarilla level touch or trend reversal. Uses discrete sizing 0.20 to minimize fee drag. 
Target 60-150 total trades over 4 years (15-37/year) on 1h timeframe. Works in bull/bear via 4h trend filter 
and volatility-based Camarilla levels that adapt to market conditions.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Get 1d data for volume spike detection
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = volume_1d > (2.0 * vol_ma_1d)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    # Calculate Camarilla levels from previous day (use 1d OHLC from previous completed day)
    # For Camarilla, we need previous day's high, low, close
    df_1d_ohlc = get_htf_data(prices, '1d')
    high_1d = df_1d_ohlc['high'].values
    low_1d = df_1d_ohlc['low'].values
    close_1d = df_1d_ohlc['close'].values
    
    # Camarilla levels: H3 = close + (high - low) * 1.1/4, L3 = close - (high - low) * 1.1/4
    # Using previous day's values (shifted by 1 to avoid look-ahead)
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d = np.roll(close_1d, 1)
    # First value will be invalid (rolled from last), but alignment will handle this
    camarilla_high = prev_close_1d + (prev_high_1d - prev_low_1d) * 1.1 / 4
    camarilla_low = prev_close_1d - (prev_high_1d - prev_low_1d) * 1.1 / 4
    
    # Align Camarilla levels to 1h timeframe (completed daily bar only)
    camarilla_high_aligned = align_htf_to_ltf(prices, df_1d_ohlc, camarilla_high)
    camarilla_low_aligned = align_htf_to_ltf(prices, df_1d_ohlc, camarilla_low)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA50 (50), volume MA (20), and Camarilla (need previous day)
    start_idx = max(50, 20, 1)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(volume_spike_1d_aligned[i]) or
            np.isnan(camarilla_high_aligned[i]) or np.isnan(camarilla_low_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: 08-20 UTC only
        hour = pd.Timestamp(prices['open_time'].iloc[i]).hour
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above H3 AND above 4h EMA50 AND volume spike
            long_setup = (close[i] > camarilla_high_aligned[i]) and \
                         (close[i] > ema_50_4h_aligned[i]) and \
                         volume_spike_1d_aligned[i]
            # Short: price breaks below L3 AND below 4h EMA50 AND volume spike
            short_setup = (close[i] < camarilla_low_aligned[i]) and \
                          (close[i] < ema_50_4h_aligned[i]) and \
                          volume_spike_1d_aligned[i]
            
            if long_setup:
                signals[i] = 0.20
                position = 1
            elif short_setup:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.20
            # Exit: price touches L3 OR closes below 4h EMA50
            if (close[i] <= camarilla_low_aligned[i]) or \
               (close[i] < ema_50_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.20
            # Exit: price touches H3 OR closes above 4h EMA50
            if (close[i] >= camarilla_high_aligned[i]) or \
               (close[i] > ema_50_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_Camarilla_H3L3_Breakout_4hTrend_1dVolSpike"
timeframe = "1h"
leverage = 1.0