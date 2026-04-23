#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R reversal with 1d EMA50 trend filter and volume spike confirmation.
Long when Williams %R crosses above -80 from oversold AND price > 1d EMA50 AND volume > 1.5x 20-period MA.
Short when Williams %R crosses below -20 from overbought AND price < 1d EMA50 AND volume > 1.5x 20-period MA.
Exit when Williams %R reaches opposite extreme (-20 for long, -80 for short) or EMA50 reverses.
Williams %R captures momentum reversals, 1d EMA50 filters major trend, volume spike confirms conviction.
Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
Works in bull (buy pullbacks in uptrend) and bear (sell rallies in downtrend).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Williams %R (14-period)
    lookback = 14
    williams_r = np.full(n, np.nan)
    
    for i in range(lookback - 1, n):
        highest_high = np.max(high[i-lookback+1:i+1])
        lowest_low = np.min(low[i-lookback+1:i+1])
        if highest_high == lowest_low:
            williams_r[i] = -50.0  # avoid division by zero
        else:
            williams_r[i] = -100.0 * (highest_high - close[i]) / (highest_high - lowest_low)
    
    # Calculate 1d EMA50 for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate volume MA (20-period) for spike filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(lookback - 1, 50, 20)  # Williams %R, EMA50, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r[i]) or np.isnan(ema_50_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        wr = williams_r[i]
        ema_val = ema_50_aligned[i]
        vol_ma_val = vol_ma_20[i]
        
        # Calculate Williams %R cross for reversal signals
        if i >= start_idx + 1:
            wr_prev = williams_r[i-1]
            wr_cross_above_80 = wr_prev <= -80.0 and wr > -80.0
            wr_cross_below_20 = wr_prev >= -20.0 and wr < -20.0
        else:
            wr_cross_above_80 = False
            wr_cross_below_20 = False
        
        # Volume filter: 6h volume > 1.5x 20-period MA
        vol_filter = volume[i] > 1.5 * vol_ma_val
        
        if position == 0:
            # Long: Williams %R crosses above -80 from oversold AND price > EMA50 AND volume filter
            if wr_cross_above_80 and price > ema_val and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 from overbought AND price < EMA50 AND volume filter
            elif wr_cross_below_20 and price < ema_val and vol_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: Williams %R reaches -20 (overbought) OR EMA50 starts falling
                if wr >= -20.0 or (i >= start_idx + 1 and ema_val < ema_50_aligned[i-1]):
                    exit_signal = True
            elif position == -1:
                # Short exit: Williams %R reaches -80 (oversold) OR EMA50 starts rising
                if wr <= -80.0 or (i >= start_idx + 1 and ema_val > ema_50_aligned[i-1]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_WilliamsR_Reversal_1dEMA50_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0