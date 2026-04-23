#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R Mean Reversion with 1w trend filter and volume spike confirmation.
Long when Williams %R(14) crosses above -80 (oversold) AND price > 1w EMA50 (uptrend) AND volume > 2.0x average.
Short when Williams %R(14) crosses below -20 (overbought) AND price < 1w EMA50 (downtrend) AND volume > 2.0x average.
Uses 6h timeframe to target 50-150 total trades over 4 years. Williams %R identifies exhaustion points in both bull and bear markets.
1w EMA50 filter ensures trades align with higher timeframe trend, reducing counter-trend whipsaws.
Volume spike confirms conviction at reversal points. Works in ranging and trending markets by fading extremes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 70:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data for EMA50 trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA50
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF indicators to 6h timeframe
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate Williams %R(14) on primary timeframe
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # Volume average (20-period) on primary timeframe
    vol_ma_primary = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(70, n):  # Start after warmup period
        # Skip if data not ready
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(vol_ma_primary[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema50_val = ema50_1w_aligned[i]
        wr = williams_r[i]
        vol_ma_val = vol_ma_primary[i]
        
        # Get current price and volume
        price = close[i]
        vol_current = volume[i]
        
        if position == 0:
            # Long: Williams %R crosses above -80 (oversold) AND price > 1w EMA50 (uptrend) AND volume spike
            if (wr > -80 and williams_r[i-1] <= -80 and price > ema50_val and vol_current > 2.0 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 (overbought) AND price < 1w EMA50 (downtrend) AND volume spike
            elif (wr < -20 and williams_r[i-1] >= -20 and price < ema50_val and vol_current > 2.0 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Williams %R crosses above -20 (overbought) OR price crosses below 1w EMA50 (trend reversal)
                if (wr > -20 and williams_r[i-1] <= -20) or price < ema50_val:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Williams %R crosses below -80 (oversold) OR price crosses above 1w EMA50 (trend reversal)
                if (wr < -80 and williams_r[i-1] >= -80) or price > ema50_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_WilliamsR_MeanReversion_1wEMA50_VolumeSpike"
timeframe = "6h"
leverage = 1.0