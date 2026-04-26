#!/usr/bin/env python3
"""
6h_Camarilla_R3_S3_Breakout_1wTrend_VolumeSpike_v1
Hypothesis: 6h Camarilla R3/S3 breakout with 1w EMA50 trend filter and volume spike confirmation.
- Uses 6h timeframe targeting 50-150 total trades over 4 years (12-37/year)
- Long when price breaks above R3 with volume spike and 1w uptrend (close > EMA50)
- Short when price breaks below S3 with volume spike and 1w downtrend (close < EMA50)
- Camarilla levels derived from previous 1d OHLC for structure-aware entries
- Volume spike confirms institutional participation (2x 20-period average)
- Weekly trend filter reduces whipsaw in bear markets (2022) and captures major moves
- Designed for low trade frequency with proven edge on BTC/ETH from historical data
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:  # Need enough data for calculations
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels from previous 1d bar
    # Camarilla: R3 = C + (H-L)*1.1/4, S3 = C - (H-L)*1.1/4
    prev_close = df_1d['close'].values
    prev_high = df_1d['high'].values
    prev_low = df_1d['low'].values
    
    R3 = prev_close + (prev_high - prev_low) * 1.1 / 4
    S3 = prev_close - (prev_high - prev_low) * 1.1 / 4
    
    # Align Camarilla levels to 6h timeframe (wait for completed 1d bar)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    
    # Load 1w data ONCE before loop for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w EMA50 for trend filter
    ema50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate volume spike (20-period volume average on 6h)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma20 * 2.0)  # Volume at least 2x average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 50 for EMA50, 20 for volume MA)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Camarilla breakout conditions with volume confirmation and trend filter
        price_above_R3 = close[i] > R3_aligned[i]
        price_below_S3 = close[i] < S3_aligned[i]
        
        # 1w trend filter
        trend_up = close[i] > ema50_1w_aligned[i]
        trend_down = close[i] < ema50_1w_aligned[i]
        
        if position == 0:
            # Long: price breaks above R3 AND volume spike AND 1w uptrend
            if price_above_R3 and volume_spike[i] and trend_up:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 AND volume spike AND 1w downtrend
            elif price_below_S3 and volume_spike[i] and trend_down:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price falls below S3 OR 1w trend turns down
            if price_below_S3 or not trend_up:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price rises above R3 OR 1w trend turns up
            if price_above_R3 or not trend_down:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Camarilla_R3_S3_Breakout_1wTrend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0