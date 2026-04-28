#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolumeS
Hypothesis: On 1h timeframe, trade breakouts of 1-day Camarilla R1/S1 levels with confirmation from 4h trend (EMA50) and 1-day volume spike.
Designed to capture momentum moves in both bull and bear markets by aligning with higher timeframe trend and filtering with volume.
Target: 15-37 trades per year (60-150 total over 4 years) to minimize fee drag while capturing meaningful moves.
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
    
    # Get 1-day data for Camarilla pivots and volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 1-day volume spike (>2.0x 20-period MA)
    vol_ma_20_1d = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = df_1d['volume'].values > (2.0 * vol_ma_20_1d)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    
    # Calculate Camarilla levels from previous 1-day bar (typical price method)
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    range_1d = df_1d['high'] - df_1d['low']
    
    # Camarilla levels
    R1 = typical_price + (range_1d * 1.1 / 12)
    S1 = typical_price - (range_1d * 1.1 / 12)
    
    # Align Camarilla levels to 1h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1.values)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or 
            np.isnan(ema_50_4h_aligned[i]) or np.isnan(vol_spike_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend direction from 4h EMA50
        trend_up = close[i] > ema_50_4h_aligned[i]
        trend_down = close[i] < ema_50_4h_aligned[i]
        
        # Volume confirmation from 1-day
        vol_confirm = vol_spike_1d_aligned[i]
        
        # Price relative to Camarilla levels
        price_above_R1 = close[i] > R1_aligned[i]
        price_below_S1 = close[i] < S1_aligned[i]
        
        # Entry logic:
        # Long: Breakout above R1 in uptrend with volume spike
        long_entry = vol_confirm and trend_up and price_above_R1 and (close[i-1] <= R1_aligned[i-1])
        
        # Short: Breakdown below S1 in downtrend with volume spike
        short_entry = vol_confirm and trend_down and price_below_S1 and (close[i-1] >= S1_aligned[i-1])
        
        # Exit logic: Opposite level or trend reversal
        long_exit = price_below_S1 or not trend_up
        short_exit = price_above_R1 or not trend_down
        
        if long_entry and position <= 0:
            signals[i] = 0.20
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.20
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolumeS"
timeframe = "1h"
leverage = 1.0