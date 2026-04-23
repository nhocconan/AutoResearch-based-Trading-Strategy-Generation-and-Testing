#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R reversal with 1d EMA34 trend filter and volume confirmation.
Long when Williams %R crosses above -80 from below AND price > 1d EMA34 AND volume > 1.3x 20-period MA.
Short when Williams %R crosses below -20 from above AND price < 1d EMA34 AND volume > 1.3x 20-period MA.
Exit when Williams %R reaches opposite extreme (-20 for long, -80 for short) or EMA34 reverses.
Williams %R identifies overbought/oversold conditions, 1d EMA34 filters higher timeframe trend,
volume confirms reversal strength. Designed to work in ranging and trending markets by
fading extremes in the direction of the 1d trend.
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
    
    # Calculate 6h Williams %R (14-period)
    williams_r = np.full(n, np.nan)
    for i in range(14, n):
        highest_high = np.max(high[i-14:i+1])
        lowest_low = np.min(low[i-14:i+1])
        if highest_high == lowest_low:
            williams_r[i] = -50.0  # avoid division by zero
        else:
            williams_r[i] = -100 * (highest_high - close[i]) / (highest_high - lowest_low)
    
    # Calculate 1d EMA34 for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 6h volume MA (20-period) for spike filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(14, 34, 20)  # Williams %R (needs 14), EMA34, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r[i]) or np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        wr = williams_r[i]
        ema_val = ema_34_aligned[i]
        vol_ma_val = vol_ma_20[i]
        
        # Calculate Williams %R crossover signals
        wr_prev = williams_r[i-1] if i > 0 else np.nan
        wr_cross_above_80 = (wr_prev <= -80) and (wr > -80)
        wr_cross_below_20 = (wr_prev >= -20) and (wr < -20)
        
        # Volume filter: 6h volume > 1.3x 20-period MA
        vol_filter = volume[i] > 1.3 * vol_ma_val
        
        if position == 0:
            # Long: Williams %R crosses above -80 AND price > 1d EMA34 AND volume filter
            if wr_cross_above_80 and price > ema_val and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 AND price < 1d EMA34 AND volume filter
            elif wr_cross_below_20 and price < ema_val and vol_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: Williams %R reaches -20 OR price < 1d EMA34
                if wr >= -20 or price < ema_val:
                    exit_signal = True
            elif position == -1:
                # Short exit: Williams %R reaches -80 OR price > 1d EMA34
                if wr <= -80 or price > ema_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_WilliamsR_Reversal_1dEMA34_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0