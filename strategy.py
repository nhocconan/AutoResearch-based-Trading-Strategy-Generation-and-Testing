#!/usr/bin/env python3
"""
1h_Camarilla_R1S1_Breakout_4hEMA34_Trend_VolumeSpike
Hypothesis: Trade 1h timeframe using Camarilla pivot levels (R1, S1) from prior 4h for entry,
4h EMA34 for trend filter, and 1h volume spike (>2.0x 20-bar MA) for confirmation.
Enter long when price breaks above Camarilla R1 AND above 4h EMA34 AND volume spike.
Enter short when price breaks below Camarilla S1 AND below 4h EMA34 AND volume spike.
Exit on opposite Camarilla touch (S1 for long, R1 for short) or trend reversal.
Uses discrete sizing 0.20 to balance return and drawdown. Target 15-35 trades/year on 1h timeframe.
Camarilla R1/S1 levels provide breakout points with moderate filtering.
The 4h EMA34 filter ensures we only trade with the 4h trend, improving performance in both bull and bear markets.
Volume confirmation avoids breakouts from low participation.
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
    
    # Get 4h data for Camarilla pivot levels (prior 4h bar)
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Camarilla levels for prior 4h: R1, S1
    # Camarilla: R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    camarilla_r1_4h = close_4h + (1.1 * (high_4h - low_4h) / 12)
    camarilla_s1_4h = close_4h - (1.1 * (high_4h - low_4h) / 12)
    
    # Align Camarilla levels to 1h timeframe (prior 4h bar's levels available at 4h close)
    camarilla_r1_4h_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r1_4h)
    camarilla_s1_4h_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s1_4h)
    
    # Get 4h data for EMA34 trend filter
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Calculate 20-bar volume MA on 1h for volume spike detection
    vol_ma_1h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike_1h = volume > (2.0 * vol_ma_1h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for 4h EMA34 (34) and 1h volume MA (20)
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r1_4h_aligned[i]) or np.isnan(camarilla_s1_4h_aligned[i]) or
            np.isnan(ema_34_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: 08-20 UTC
        hour = pd.Timestamp(prices.iloc[i]['open_time']).hour
        in_session = 8 <= hour <= 20
        
        if not in_session:
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above Camarilla R1 AND above 4h EMA34 AND volume spike
            long_setup = (close[i] > camarilla_r1_4h_aligned[i]) and \
                         (close[i] > ema_34_4h_aligned[i]) and \
                         volume_spike_1h[i]
            # Short: price breaks below Camarilla S1 AND below 4h EMA34 AND volume spike
            short_setup = (close[i] < camarilla_s1_4h_aligned[i]) and \
                          (close[i] < ema_34_4h_aligned[i]) and \
                          volume_spike_1h[i]
            
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
            # Exit: price touches Camarilla S1 OR closes below 4h EMA34
            if (close[i] <= camarilla_s1_4h_aligned[i]) or \
               (close[i] < ema_34_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.20
            # Exit: price touches Camarilla R1 OR closes above 4h EMA34
            if (close[i] >= camarilla_r1_4h_aligned[i]) or \
               (close[i] > ema_34_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_Camarilla_R1S1_Breakout_4hEMA34_Trend_VolumeSpike"
timeframe = "1h"
leverage = 1.0