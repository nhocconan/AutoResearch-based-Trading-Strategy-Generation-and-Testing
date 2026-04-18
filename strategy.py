#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R with 1d trend filter and volume confirmation.
# Williams %R measures overbought/oversold levels. Long when %R crosses above -80 from below (oversold bounce).
# Short when %R crosses below -20 from above (overbought reversal).
# Uses 1d EMA(50) as trend filter: only long when price > EMA, only short when price < EMA.
# Volume filter: current volume > 1.5x 24-period average (24 * 1h = 1 day equivalent).
# Designed for ~15-30 trades/year per symbol with low turnover to minimize fee drag.
name = "12h_WilliamsR_1dEMA50_Volume_Filter"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # EMA(50) on 1d close for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams %R (14-period) on 12h data
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when high == low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Volume filter: current volume > 1.5 * 24-period average (24 * 1h = 1 day)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_filter = volume > (1.5 * vol_ma_24)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(williams_r[i]) or
            np.isnan(vol_ma_24[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema_val = ema_50_1d_aligned[i]
        wr = williams_r[i]
        vol_filter = volume_filter[i]
        
        if position == 0:
            # Long: Williams %R crosses above -80 from below (bullish reversal) with trend and volume
            if wr > -80 and williams_r[i-1] <= -80 and close_val > ema_val and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 from above (bearish reversal) with trend and volume
            elif wr < -20 and williams_r[i-1] >= -20 and close_val < ema_val and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Williams %R crosses below -50 (momentum fading) or trend fails
            if wr < -50 or close_val < ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Williams %R crosses above -50 (momentum fading) or trend fails
            if wr > -50 or close_val > ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals