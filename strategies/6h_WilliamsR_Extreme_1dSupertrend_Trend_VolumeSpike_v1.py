#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R Extreme + 1d Supertrend Trend Filter + Volume Spike.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d Supertrend (ATR=10, mult=3) for trend filter (price > Supertrend = uptrend, price < Supertrend = downtrend).
- Entry: Long when Williams %R(14) crosses above -20 (exiting oversold) AND price > 1d Supertrend AND volume > 2.0 * 6h volume MA(20);
         Short when Williams %R(14) crosses below -80 (exiting overbought) AND price < 1d Supertrend AND volume > 2.0 * 6h volume MA(20).
- Exit: Long exits when Williams %R crosses below -80; Short exits when Williams %R crosses above -20.
- Signal size: 0.25 discrete to balance capture and fee control.
- Williams %R captures momentum exhaustion/reversal; Supertrend filters higher-timeframe trend; volume spike confirms conviction.
- Works in bull (buying oversold bounces in uptrend) and bear (selling overbought rejections in downtrend) with reduced whipsaws.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Supertrend trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate 1d Supertrend (ATR=10, mult=3)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # first period
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR
    atr = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # Basic Upper and Lower Bands
    basic_ub = (high_1d + low_1d) / 2 + 3.0 * atr
    basic_lb = (high_1d + low_1d) / 2 - 3.0 * atr
    
    # Supertrend
    supertrend = np.zeros_like(close_1d)
    direction = np.ones_like(close_1d)  # 1 for uptrend, -1 for downtrend
    
    supertrend[0] = basic_lb[0]
    direction[0] = 1
    
    for i in range(1, len(close_1d)):
        if close_1d[i] > supertrend[i-1]:
            supertrend[i] = basic_ub[i]
            direction[i] = 1
        else:
            supertrend[i] = basic_lb[i]
            direction[i] = -1
    
    # Align Supertrend and direction to 6h timeframe
    supertrend_aligned = align_htf_to_ltf(prices, df_1d, supertrend)
    direction_aligned = align_htf_to_ltf(prices, df_1d, direction)
    
    # Williams %R on 6h (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # Get 6h data for volume MA(20)
    vol_ma_6h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(14, 10)  # Williams %R needs 14, Supertrend needs 10
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(supertrend_aligned[i]) or np.isnan(direction_aligned[i]) or 
            np.isnan(williams_r[i]) or np.isnan(vol_ma_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        prev_williams_r = williams_r[i-1] if i > 0 else -50
        
        # Trend filter from 1d Supertrend
        uptrend = direction_aligned[i] == 1
        downtrend = direction_aligned[i] == -1
        
        # Volume confirmation: 2.0x threshold
        vol_confirm = curr_volume > 2.0 * vol_ma_6h[i]
        
        if position == 0:
            # Check for entry signals
            if uptrend and vol_confirm:
                # Long: Williams %R crosses above -20 (exiting oversold)
                if prev_williams_r <= -20 and williams_r[i] > -20:
                    signals[i] = 0.25
                    position = 1
            elif downtrend and vol_confirm:
                # Short: Williams %R crosses below -80 (exiting overbought)
                if prev_williams_r >= -80 and williams_r[i] < -80:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long position: exit when Williams %R crosses below -80
            if prev_williams_r > -80 and williams_r[i] <= -80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit when Williams %R crosses above -20
            if prev_williams_r < -20 and williams_r[i] >= -20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_Extreme_1dSupertrend_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0