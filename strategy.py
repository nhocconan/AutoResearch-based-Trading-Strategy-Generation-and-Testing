#!/usr/bin/env python3
"""
Hypothesis: 4h Williams %R mean reversion with 1d EMA50 trend filter and volume spike confirmation.
Long when Williams %R < -80 (oversold) AND price > 1d EMA50 (uptrend) AND volume > 1.8x average.
Short when Williams %R > -20 (overbought) AND price < 1d EMA50 (downtrend) AND volume > 1.8x average.
Exit when Williams %R crosses above -50 (long) or below -50 (short).
Uses 4h timeframe with mean reversion logic to capture bounces in ranging markets and pullbacks in trends.
Williams %R is effective in both bull and bear markets as it identifies exhaustion points.
Target: 60-120 trades over 4 years (15-30/year) to stay within proven working range.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 4h Williams %R (14-period) - ONCE before loop
    period = 14
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    # Avoid division by zero
    denominator = highest_high - lowest_low
    williams_r = np.where(denominator != 0, -100 * (highest_high - close) / denominator, -50)
    
    # Load 1d data for EMA50 trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF indicators to 4h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, prices, williams_r)  # Same timeframe, no alignment needed but keep for consistency
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume average (20-period) on primary timeframe
    vol_ma_primary = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup period
        # Skip if data not ready
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma_primary[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        wr_val = williams_r_aligned[i]
        ema50_val = ema50_1d_aligned[i]
        vol_ma_val = vol_ma_primary[i]
        
        # Get current price and volume
        price = close[i]
        vol_current = volume[i]
        
        if position == 0:
            # Long: Williams %R oversold (< -80) AND price > 1d EMA50 (uptrend) AND volume spike
            if (wr_val < -80.0 and price > ema50_val and vol_current > 1.8 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20) AND price < 1d EMA50 (downtrend) AND volume spike
            elif (wr_val > -20.0 and price < ema50_val and vol_current > 1.8 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: Williams %R crosses -50 (mean reversion midpoint)
            exit_signal = False
            
            if position == 1:
                # Exit long: Williams %R crosses above -50
                if wr_val > -50.0:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Williams %R crosses below -50
                if wr_val < -50.0:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_WilliamsR_MeanReversion_1dEMA50_Volume"
timeframe = "4h"
leverage = 1.0