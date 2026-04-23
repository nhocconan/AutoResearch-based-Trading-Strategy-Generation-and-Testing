#!/usr/bin/env python3
"""
Hypothesis: 4h Williams %R reversal with 1d EMA34 trend filter and volume spike confirmation.
Long when Williams %R(14) crosses above -80 (oversold reversal) AND 1d EMA34 rising AND 1h volume > 1.8x 20-period MA.
Short when Williams %R(14) crosses below -20 (overbought reversal) AND 1d EMA34 falling AND 1h volume > 1.8x 20-period MA.
Exit when Williams %R crosses opposite threshold (-20 for long, -80 for short) or 1d EMA34 reverses.
Uses 1d HTF for trend filter to avoid counter-trend trades, volume spike for momentum confirmation.
Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.
Williams %R provides timely reversal signals in ranging markets, effective in both bull and bear regimes.
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
    
    # Calculate Williams %R(14) - momentum oscillator
    # %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    williams_r = np.full(n, np.nan)
    lookback = 14
    
    for i in range(lookback - 1, n):
        highest_high = np.max(high[i - lookback + 1:i + 1])
        lowest_low = np.min(low[i - lookback + 1:i + 1])
        if highest_high != lowest_low:
            williams_r[i] = (highest_high - close[i]) / (highest_high - lowest_low) * -100
        else:
            williams_r[i] = -50  # neutral when range is zero
    
    # Calculate 1d EMA34 for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1h volume MA (20-period) for spike filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(lookback, 34, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r[i]) or np.isnan(ema_34_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        wr = williams_r[i]
        ema_val = ema_34_aligned[i]
        vol_ma_val = vol_ma_20[i]
        
        # Calculate Williams %R crossover signals
        wr_prev = williams_r[i-1]
        wr_cross_above_80 = wr_prev <= -80 and wr > -80   # oversold reversal
        wr_cross_below_20 = wr_prev >= -20 and wr < -20   # overbought reversal
        
        # Calculate EMA34 slope for trend direction (rising/falling)
        if i >= start_idx + 1:
            ema_prev = ema_34_aligned[i-1]
            ema_rising = ema_val > ema_prev
            ema_falling = ema_val < ema_prev
        else:
            ema_rising = False
            ema_falling = False
        
        # Volume filter: 1h volume > 1.8x 20-period MA (adaptive to volatility)
        vol_filter = volume[i] > 1.8 * vol_ma_val
        
        if position == 0:
            # Long: Williams %R crosses above -80 (oversold reversal) AND EMA34 rising AND volume filter
            if wr_cross_above_80 and ema_rising and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 (overbought reversal) AND EMA34 falling AND volume filter
            elif wr_cross_below_20 and ema_falling and vol_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: Williams %R crosses below -20 (overbought) OR EMA34 starts falling
                if wr_cross_below_20 or (i >= start_idx + 1 and ema_val < ema_34_aligned[i-1]):
                    exit_signal = True
            elif position == -1:
                # Short exit: Williams %R crosses above -80 (oversold) OR EMA34 starts rising
                if wr_cross_above_80 or (i >= start_idx + 1 and ema_val > ema_34_aligned[i-1]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_WilliamsR_Reversal_1dEMA34_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0