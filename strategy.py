#!/usr/bin/env python3
"""
Hypothesis: 4h Williams %R reversal with 1d EMA50 trend filter and volume confirmation.
Long when Williams %R crosses above -80 from below AND price > 1d EMA50 AND volume > 1.5x 20-period MA.
Short when Williams %R crosses below -20 from above AND price < 1d EMA50 AND volume > 1.5x 20-period MA.
Exit when Williams %R reaches opposite extreme (-20 for long, -80 for short) or trend reverses.
Uses 1d HTF for trend filter to avoid counter-trend trades, Williams %R for mean reversion in extremes,
volume confirmation to ensure momentum. Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.
Williams %R identifies overbought/oversold conditions, effective in both bull (pullbacks) and bear (bounces) markets.
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
    
    # Calculate 1d EMA50 for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate volume MA (20-period) for confirmation
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
        
        # Calculate Williams %R crossover signals
        wr_long_signal = False
        wr_short_signal = False
        if i >= start_idx + 1:
            wr_prev = williams_r[i-1]
            # Long: Williams %R crosses above -80 from below
            if wr_prev <= -80 and wr > -80:
                wr_long_signal = True
            # Short: Williams %R crosses below -20 from above
            if wr_prev >= -20 and wr < -20:
                wr_short_signal = True
        
        # Volume filter: current volume > 1.5x 20-period MA
        vol_filter = volume[i] > 1.5 * vol_ma_val
        
        # Trend filter: price relative to 1d EMA50
        price_above_ema = price > ema_val
        price_below_ema = price < ema_val
        
        if position == 0:
            # Long: Williams %R crosses above -80 AND price > 1d EMA50 AND volume filter
            if wr_long_signal and price_above_ema and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 AND price < 1d EMA50 AND volume filter
            elif wr_short_signal and price_below_ema and vol_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: Williams %R reaches -20 (overbought) OR price < 1d EMA50 (trend reverse)
                if wr >= -20 or price < ema_val:
                    exit_signal = True
            elif position == -1:
                # Short exit: Williams %R reaches -80 (oversold) OR price > 1d EMA50 (trend reverse)
                if wr <= -80 or price > ema_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_WilliamsR_Reversal_1dEMA50_Trend_VolumeConfirm"
timeframe = "4h"
leverage = 1.0