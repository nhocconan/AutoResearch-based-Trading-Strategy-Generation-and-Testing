#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_12hTrend_VolumeSpike_v2
Hypothesis: 4h Camarilla R1/S1 breakout with 12h EMA50 trend filter and volume spike confirmation.
- Uses 4h timeframe for optimal trade frequency (target: 75-200 total trades over 4 years)
- Camarilla pivot levels (R1, S1) calculated from prior 1d candles
- 12h EMA50 trend filter ensures trades align with higher timeframe trend
- Volume confirmation requires current volume > 1.5x 20-period average
- Long when price breaks above R1 AND 12h EMA50 up AND volume spike
- Short when price breaks below S1 AND 12h EMA50 down AND volume spike
- Designed for 19-50 trades/year (75-200 total over 4 years) to minimize fee drag
- Works in bull/bear markets by trading with the 12h trend and using Camarilla levels for entry/exit
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need enough data for calculations
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels (R1, S1) from prior day OHLC
    # Camarilla: R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    camarilla_r1 = close_1d + 1.1 * (high_1d - low_1d) / 12
    camarilla_s1 = close_1d - 1.1 * (high_1d - low_1d) / 12
    
    # Align Camarilla levels to 4h timeframe (completed 1d bar only)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Load 12h data ONCE before loop for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 20 for volume MA, 50 for 12h EMA)
    start_idx = max(20, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(ema50_12h_aligned[i]) or np.isnan(vol_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Camarilla breakout conditions
        price_above_r1 = close[i] > camarilla_r1_aligned[i]
        price_below_s1 = close[i] < camarilla_s1_aligned[i]
        
        # 12h EMA50 trend filter
        ema50_uptrend = close[i] > ema50_12h_aligned[i]
        ema50_downtrend = close[i] < ema50_12h_aligned[i]
        
        # Volume confirmation
        vol_confirm = volume_spike[i]
        
        if position == 0:
            # Long: price breaks above R1 AND 12h EMA50 up AND volume spike
            if price_above_r1 and ema50_uptrend and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 AND 12h EMA50 down AND volume spike
            elif price_below_s1 and ema50_downtrend and vol_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price falls below S1 OR 12h EMA50 turns down
            if price_below_s1 or not ema50_uptrend:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price rises above R1 OR 12h EMA50 turns up
            if price_above_r1 or not ema50_downtrend:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_12hTrend_VolumeSpike_v2"
timeframe = "4h"
leverage = 1.0