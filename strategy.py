#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R Extreme + 1d EMA50 Trend Filter + Volume Spike.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d EMA50 for trend filter (price > EMA50 = uptrend, price < EMA50 = downtrend).
- Entry: Long when Williams %R(14) < -80 (oversold) AND price > 1d EMA50 AND volume > 2.0 * 6h volume MA(20);
         Short when Williams %R(14) > -20 (overbought) AND price < 1d EMA50 AND volume > 2.0 * 6h volume MA(20).
- Exit: Long exits when Williams %R(14) crosses above -50; Short exits when Williams %R(14) crosses below -50.
- Signal size: 0.25 discrete to balance capture and fee control.
- Works in bull (buying oversold dips in uptrend) and bear (selling overbought rallies in downtrend).
- Uses 1d Williams %R and EMA50 applied to 6h chart with proper MTF alignment.
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
    
    # Get 1d data for Williams %R and EMA50
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:  # Need at least 14 days for Williams %R
        return np.zeros(n)
    
    # Calculate 1d Williams %R(14): (Highest High - Close) / (Highest High - Lowest Low) * -100
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Rolling window for Williams %R
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_1d) / (highest_high - lowest_low)
    
    # Calculate 1d EMA50
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d indicators to 6h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Get 6h data for volume MA(20)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 14)  # EMA50 needs 50, volume MA needs 20, Williams %R needs 14
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_aligned[i]) or 
            np.isnan(williams_r_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        curr_williams_r = williams_r_aligned[i]
        
        # Volume confirmation: 2.0x threshold
        vol_confirm = curr_volume > 2.0 * vol_ma[i]
        
        if position == 0:
            # Check for entry signals
            if vol_confirm:
                # Long: Williams %R < -80 (oversold) AND price > 1d EMA50 (uptrend)
                if curr_williams_r < -80 and curr_close > ema_50_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: Williams %R > -20 (overbought) AND price < 1d EMA50 (downtrend)
                elif curr_williams_r > -20 and curr_close < ema_50_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long position: exit when Williams %R crosses above -50
            if curr_williams_r > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit when Williams %R crosses below -50
            if curr_williams_r < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_Extreme_1dEMA50_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0