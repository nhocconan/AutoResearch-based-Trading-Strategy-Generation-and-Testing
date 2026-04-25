#!/usr/bin/env python3
"""
1d_Camarilla_H3L3_Breakout_1wTrend_VolumeSpike
Hypothesis: Trade 1d timeframe using weekly Camarilla H3/L3 breakout for entry,
weekly EMA34 for trend filter, and daily volume spike (>2.0x 20-bar MA) for confirmation.
Enter long when price breaks above weekly Camarilla H3 AND above weekly EMA34 AND volume spike.
Enter short when price breaks below weekly Camarilla L3 AND below weekly EMA34 AND volume spike.
Exit on opposite Camarilla touch (L3 for long, H3 for short) or trend reversal.
Uses discrete sizing 0.25 to balance return and drawdown. Target 7-25 trades/year on 1d timeframe.
Works in bull/bear via weekly structure and trend filter.
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
    
    # Get 1w data for weekly Camarilla pivot levels and EMA34
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly Camarilla pivot levels (based on previous week)
    # Camarilla: H3 = close + 1.1*(high-low)/2, L3 = close - 1.1*(high-low)/2
    # Using previous week's OHLC to avoid look-ahead
    prev_week_high = np.roll(high_1w, 1)
    prev_week_low = np.roll(low_1w, 1)
    prev_week_close = np.roll(close_1w, 1)
    prev_week_high[0] = np.nan  # first week has no previous
    prev_week_low[0] = np.nan
    prev_week_close[0] = np.nan
    
    camarilla_h3 = prev_week_close + 1.1 * (prev_week_high - prev_week_low) / 2
    camarilla_l3 = prev_week_close - 1.1 * (prev_week_high - prev_week_low) / 2
    
    # Align weekly Camarilla levels to 1d timeframe (completed weekly bar only)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l3)
    
    # Calculate weekly EMA34 for trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Get 1d data for daily volume spike confirmation
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    
    # Calculate 20-bar volume MA on 1d for volume spike detection
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = volume_1d > (2.0 * vol_ma_1d)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for Camarilla (needs prev week), EMA34 (34), volume MA (20)
    start_idx = max(34, 20) + 1  # +1 for roll shift
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(volume_spike_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above weekly Camarilla H3 AND above weekly EMA34 AND volume spike
            long_setup = (close[i] > camarilla_h3_aligned[i]) and \
                         (close[i] > ema_34_1w_aligned[i]) and \
                         volume_spike_1d_aligned[i]
            # Short: price breaks below weekly Camarilla L3 AND below weekly EMA34 AND volume spike
            short_setup = (close[i] < camarilla_l3_aligned[i]) and \
                          (close[i] < ema_34_1w_aligned[i]) and \
                          volume_spike_1d_aligned[i]
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price touches weekly Camarilla L3 OR closes below weekly EMA34
            if (close[i] <= camarilla_l3_aligned[i]) or \
               (close[i] < ema_34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price touches weekly Camarilla H3 OR closes above weekly EMA34
            if (close[i] >= camarilla_h3_aligned[i]) or \
               (close[i] > ema_34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Camarilla_H3L3_Breakout_1wTrend_VolumeSpike"
timeframe = "1d"
leverage = 1.0