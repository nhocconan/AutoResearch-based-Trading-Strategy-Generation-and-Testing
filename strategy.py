#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R reversal with 12h EMA50 trend filter and 1d volume spike confirmation.
Long when Williams %R crosses above -80 from oversold AND 12h EMA50 rising AND 1d volume > 1.5x 20-period MA.
Short when Williams %R crosses below -20 from overbought AND 12h EMA50 falling AND 1d volume > 1.5x 20-period MA.
Exit when Williams %R crosses opposite threshold (-20 for long exit, -80 for short exit) or 12h EMA50 reverses.
Williams %R captures mean reversion in bear market rallies and pullbacks in bull trends.
12h EMA50 filters major trend to avoid counter-trend trades.
1d volume spike confirms momentum behind the reversal.
Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
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
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    williams_r = np.full(n, np.nan)
    
    for i in range(lookback - 1, n):
        highest_high[i] = np.max(high[i-lookback+1:i+1])
        lowest_low[i] = np.min(low[i-lookback+1:i+1])
        if highest_high[i] != lowest_low[i]:
            williams_r[i] = -100 * (highest_high[i] - close[i]) / (highest_high[i] - lowest_low[i])
        else:
            williams_r[i] = -50  # neutral when range is zero
    
    # Calculate 12h EMA50 for trend filter (HTF)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 1d volume MA (20-period) for spike filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(lookback - 1, 50, 20)  # Williams %R, EMA50, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r[i]) or np.isnan(ema_50_aligned[i]) or 
            np.isnan(vol_ma_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        wr = williams_r[i]
        ema_val = ema_50_aligned[i]
        vol_ma_val = vol_ma_aligned[i]
        
        # Calculate Williams %R crossover signals
        if i >= start_idx + 1:
            wr_prev = williams_r[i-1]
            wr_cross_above_80 = wr_prev <= -80 and wr > -80  # crossed above -80 (exit oversold)
            wr_cross_below_20 = wr_prev >= -20 and wr < -20  # crossed below -20 (exit overbought)
        else:
            wr_cross_above_80 = False
            wr_cross_below_20 = False
        
        # Calculate EMA50 slope for trend direction
        if i >= start_idx + 1:
            ema_prev = ema_50_aligned[i-1]
            ema_rising = ema_val > ema_prev
            ema_falling = ema_val < ema_prev
        else:
            ema_rising = False
            ema_falling = False
        
        # Volume filter: 1d volume > 1.5x 20-period MA
        vol_filter = volume_1d[i] > 1.5 * vol_ma_val if i < len(volume_1d) else False
        
        if position == 0:
            # Long: Williams %R crosses above -80 (exit oversold) AND EMA50 rising AND volume filter
            if wr_cross_above_80 and ema_rising and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 (exit overbought) AND EMA50 falling AND volume filter
            elif wr_cross_below_20 and ema_falling and vol_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: Williams %R crosses below -20 (enter overbought) OR EMA50 starts falling
                if i >= start_idx + 1:
                    wr_cross_below_20_exit = wr_prev >= -20 and wr < -20
                    ema_falling_exit = ema_val < ema_50_aligned[i-1]
                    if wr_cross_below_20_exit or ema_falling_exit:
                        exit_signal = True
            elif position == -1:
                # Short exit: Williams %R crosses above -80 (enter oversold) OR EMA50 starts rising
                if i >= start_idx + 1:
                    wr_cross_above_80_exit = wr_prev <= -80 and wr > -80
                    ema_rising_exit = ema_val > ema_50_aligned[i-1]
                    if wr_cross_above_80_exit or ema_rising_exit:
                        exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_WilliamsR_Reversal_12hEMA50_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0