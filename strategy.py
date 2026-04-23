#!/usr/bin/env python3
"""
Hypothesis: 4h Williams %R Extreme + 1d EMA50 Trend + Volume Spike
- Long when: Williams %R(14) crosses above -80 (oversold recovery) + price > 1d EMA50 + volume > 1.5x 20-period average
- Short when: Williams %R(14) crosses below -20 (overbought rejection) + price < 1d EMA50 + volume > 1.5x 20-period average
- Exit when: Williams %R crosses -50 (mean reversion) OR ATR-based trailing stop (2.5x ATR)
- Uses 1d EMA50 as trend filter to avoid counter-trend trades
- Volume spike reduces false signals in low momentum environments
- Williams %R is effective in ranging and trending markets for mean reversion
- Target: 20-50 trades/year (80-200 total over 4 years) to minimize fee drag on 4h timeframe
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
    
    # Calculate ATR(14) for trailing stop
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with close
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Williams %R(14): (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low + 1e-10) * -100
    
    # Williams %R signals: cross above -80 (long), cross below -20 (short)
    williams_r_long_signal = (williams_r[:-1] <= -80) & (williams_r[1:] > -80)
    williams_r_long_signal = np.concatenate([[False], williams_r_long_signal])
    williams_r_short_signal = (williams_r[:-1] >= -20) & (williams_r[1:] < -20)
    williams_r_short_signal = np.concatenate([[False], williams_r_short_signal])
    williams_r_exit = (williams_r > -50) & (williams_r < -50)  # placeholder, will use cross logic
    
    # Volume confirmation: > 1.5x 20-period average (spike filter)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Load 1d EMA50 ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    long_extreme = 0.0  # highest high since long entry
    short_extreme = 0.0  # lowest low since short entry
    
    # Start from index where all indicators are ready
    start_idx = max(20, 14, 50)  # Need 20 for volume MA, 14 for Williams %R, 50 for EMA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(vol_ma[i]) or 
            np.isnan(ema_50_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                long_extreme = 0.0
                short_extreme = 0.0
            continue
        
        # Williams %R cross signals
        wr_long_cross = williams_r_long_signal[i]
        wr_short_cross = williams_r_short_signal[i]
        wr_exit_signal = (williams_r[i-1] < -50 and williams_r[i] >= -50) or \
                         (williams_r[i-1] > -50 and williams_r[i] <= -50)
        
        # Volume spike confirmation (> 1.5x average)
        volume_spike = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: Williams %R crosses above -80 + price > 1d EMA50 + volume spike
            if wr_long_cross and close[i] > ema_50_aligned[i] and volume_spike:
                signals[i] = 0.25
                position = 1
                long_extreme = high[i]
            # Short: Williams %R crosses below -20 + price < 1d EMA50 + volume spike
            elif wr_short_cross and close[i] < ema_50_aligned[i] and volume_spike:
                signals[i] = -0.25
                position = -1
                short_extreme = low[i]
        elif position == 1:
            # Update long extreme
            long_extreme = max(long_extreme, high[i])
            
            # Exit conditions:
            # 1. Williams %R crosses above -50 (mean reversion exit)
            # 2. Price reverses 2.5x ATR from long extreme (trailing stop)
            wr_exit = (williams_r[i-1] < -50 and williams_r[i] >= -50)
            trailing_stop_long = close[i] < long_extreme - 2.5 * atr[i]
            
            if wr_exit or trailing_stop_long:
                signals[i] = 0.0
                position = 0
                long_extreme = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Update short extreme
            short_extreme = min(short_extreme, low[i])
            
            # Exit conditions:
            # 1. Williams %R crosses below -50 (mean reversion exit)
            # 2. Price reverses 2.5x ATR from short extreme (trailing stop)
            wr_exit = (williams_r[i-1] > -50 and williams_r[i] <= -50)
            trailing_stop_short = close[i] > short_extreme + 2.5 * atr[i]
            
            if wr_exit or trailing_stop_short:
                signals[i] = 0.0
                position = 0
                short_extreme = 0.0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WilliamsR_Extreme_1dEMA50_VolumeSpike_ATRStop"
timeframe = "4h"
leverage = 1.0