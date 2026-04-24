#!/usr/bin/env python3
"""
Hypothesis: 4h Williams %R Extreme + 12h EMA50 Trend Filter + Volume Spike.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 12h EMA50 for trend filter (price > EMA50 = uptrend, price < EMA50 = downtrend).
- Entry: Long when Williams %R(14) crosses above -20 (oversold bounce) AND price > 12h EMA50 AND volume > 2.0 * 4h volume MA(20);
         Short when Williams %R(14) crosses below -80 (overbought rejection) AND price < 12h EMA50 AND volume > 2.0 * 4h volume MA(20).
- Exit: Long exits when Williams %R(14) crosses below -80; Short exits when Williams %R(14) crosses above -20.
- Signal size: 0.25 discrete to balance capture and fee control.
- Williams %R captures mean reversion in overextended moves; EMA50 filters higher-timeframe trend; volume spike confirms conviction.
- Works in bull (buying dips in uptrend) and bear (selling rallies in downtrend) with reduced whipsaws.
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
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50
    close_12h = df_12h['close'].values
    ema_50 = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align EMA50 to 4h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    
    # Calculate Williams %R(14) on 4h
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero (when highest_high == lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Get 4h data for volume MA(20)
    vol_ma_4h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(14, 20, 50)  # Williams %R needs 14, volume MA needs 20, EMA50 needs 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_aligned[i]) or 
            np.isnan(williams_r[i]) or 
            np.isnan(vol_ma_4h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_williams_r = williams_r[i]
        curr_volume = volume[i]
        
        # Previous Williams %R for crossover detection
        prev_williams_r = williams_r[i-1]
        
        # Trend filter from 12h EMA50
        uptrend = curr_close > ema_50_aligned[i]
        downtrend = curr_close < ema_50_aligned[i]
        
        # Volume confirmation: 2.0x threshold
        vol_confirm = curr_volume > 2.0 * vol_ma_4h[i]
        
        if position == 0:
            # Check for entry signals
            if uptrend and vol_confirm:
                # Long: Williams %R crosses above -20 (oversold bounce)
                if prev_williams_r <= -20 and curr_williams_r > -20:
                    signals[i] = 0.25
                    position = 1
            elif downtrend and vol_confirm:
                # Short: Williams %R crosses below -80 (overbought rejection)
                if prev_williams_r >= -80 and curr_williams_r < -80:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long position: exit when Williams %R crosses below -80
            if prev_williams_r > -80 and curr_williams_r <= -80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit when Williams %R crosses above -20
            if prev_williams_r < -20 and curr_williams_r >= -20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WilliamsR_Extreme_12hEMA50_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0