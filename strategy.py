#!/usr/bin/env python3
"""
Hypothesis: 12h strategy using 1d Williams %R extremes with volume confirmation and ATR trailing stop.
Long when 1d Williams %R crosses above -80 from below AND volume > 1.5x 20-period average.
Short when 1d Williams %R crosses below -20 from above AND volume > 1.5x 20-period average.
Exit when Williams %R returns to -50 level or ATR trailing stop hit (2.5*ATR from extreme).
Uses discrete position sizing (0.25) to minimize fee churn. Designed for 12h timeframe to target 12-37 trades/year.
Works in both bull and bear markets: Williams %R identifies overbought/oversold conditions,
volume confirmation filters weak moves, ATR stop manages risk during strong trends.
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
    
    # Calculate 1d Williams %R (14-period)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_1d) / (highest_high - lowest_low)
    williams_r[highest_high == lowest_low] = -50  # avoid division by zero
    
    # Align Williams %R to 12h timeframe (no extra delay needed for Williams %R)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Volume average (20-period) on 12h timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for stoploss calculation
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first bar
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    long_stop = 0.0
    short_stop = 0.0
    
    # Start from index where all indicators are ready
    start_idx = max(20, 14)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        wr = williams_r_aligned[i]
        
        if position == 0:
            # Long: Williams %R crosses above -80 from below AND volume spike
            if i > start_idx:
                wr_prev = williams_r_aligned[i-1]
                if (wr > -80 and wr_prev <= -80 and volume[i] > 1.5 * vol_ma_val):
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                    long_stop = price - 2.5 * atr_val
            # Short: Williams %R crosses below -20 from above AND volume spike
            elif i > start_idx:
                wr_prev = williams_r_aligned[i-1]
                if (wr < -20 and wr_prev >= -20 and volume[i] > 1.5 * vol_ma_val):
                    signals[i] = -0.25
                    position = -1
                    entry_price = price
                    short_stop = price + 2.5 * atr_val
        else:
            # Exit conditions
            exit_signal = False
            
            # Primary exit: Williams %R returns to -50 level (mean reversion)
            if position == 1 and wr >= -50:
                exit_signal = True
            elif position == -1 and wr <= -50:
                exit_signal = True
            
            # ATR trailing stop: trail from extreme price
            if position == 1:
                # Update long stop to trail higher
                long_stop = max(long_stop, price - 2.5 * atr_val)
                if price < long_stop:
                    exit_signal = True
            elif position == -1:
                # Update short stop to trail lower
                short_stop = min(short_stop, price + 2.5 * atr_val)
                if price > short_stop:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                long_stop = 0.0
                short_stop = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_WilliamsR_VolumeConfirmation_ATRTrailingStop"
timeframe = "12h"
leverage = 1.0