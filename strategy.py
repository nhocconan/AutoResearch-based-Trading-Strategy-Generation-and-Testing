#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation.
Long when price breaks above R3 AND price > 1d EMA34 AND volume > 1.8x 20-period average.
Short when price breaks below S3 AND price < 1d EMA34 AND volume > 1.8x 20-period average.
Exit at H5/L5 levels (Camarilla mid-point) or ATR trailing stop (2.0*ATR from entry).
Uses 1d HTF for trend alignment. Target: 50-150 total trades over 4 years (12-37/year) with discrete sizing 0.25.
Camarilla pivot levels provide institutional support/resistance that work in both trending and ranging markets.
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
    
    # Calculate 1d EMA34 for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate ATR(14) for trailing stop
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 20-period volume average for spike filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0  # for long trailing stop
    lowest_since_entry = 0.0   # for short trailing stop
    
    # Start from index where volume MA is ready (20)
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ema_val = ema_34_1d_aligned[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        
        if position == 0:
            # Calculate Camarilla levels from previous 12h bar
            if i >= 1:
                ph = high[i-1]  # previous period high
                pl = low[i-1]   # previous period low
                pc = close[i-1] # previous period close
                
                # Camarilla levels
                range_ = ph - pl
                if range_ > 0:  # avoid division by zero
                    r3 = pc + (range_ * 1.1 / 4)  # R3 = C + (H-L)*1.1/4
                    s3 = pc - (range_ * 1.1 / 4)  # S3 = C - (H-L)*1.1/4
                    h5 = pc + (range_ * 1.1 / 2)  # H5 = C + (H-L)*1.1/2
                    l5 = pc - (range_ * 1.1 / 2)  # L5 = C - (H-L)*1.1/2
                    
                    # Long: Break above R3 with trend and volume confirmation
                    if price > r3 and price > ema_val and volume[i] > 1.8 * vol_ma_val:
                        signals[i] = 0.25
                        position = 1
                        highest_since_entry = price
                    # Short: Break below S3 with trend and volume confirmation
                    elif price < s3 and price < ema_val and volume[i] > 1.8 * vol_ma_val:
                        signals[i] = -0.25
                        position = -1
                        lowest_since_entry = price
                else:
                    # If no range, hold flat
                    if position != 0:
                        signals[i] = 0.0
                        position = 0
            else:
                # Not enough history for Camarilla calculation
                if position != 0:
                    signals[i] = 0.0
                    position = 0
        else:
            # Calculate Camarilla levels for exit (H5/L5) and update trailing stops
            if i >= 1:
                ph = high[i-1]
                pl = low[i-1]
                pc = close[i-1]
                
                range_ = ph - pl
                if range_ > 0:
                    h5 = pc + (range_ * 1.1 / 2)  # H5 = C + (H-L)*1.1/2
                    l5 = pc - (range_ * 1.1 / 2)  # L5 = C - (H-L)*1.1/2
                else:
                    h5 = pc
                    l5 = pc
            else:
                h5 = close[i]
                l5 = close[i]
            
            # Update highest/lowest since entry for trailing stop
            if position == 1:
                highest_since_entry = max(highest_since_entry, price)
            elif position == -1:
                lowest_since_entry = min(lowest_since_entry, price)
            
            # Exit conditions
            exit_signal = False
            
            # Primary exit: Price reaches H5/L5 (Camarilla mid-point)
            if position == 1 and price >= h5:
                exit_signal = True
            elif position == -1 and price <= l5:
                exit_signal = True
            
            # ATR-based trailing stop: 2.0 * ATR from highest/lowest since entry
            if position == 1 and price < highest_since_entry - 2.0 * atr_val:
                exit_signal = True
            elif position == -1 and price > lowest_since_entry + 2.0 * atr_val:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeConfirmation_H5L5Exit_ATRTrailingStop"
timeframe = "12h"
leverage = 1.0