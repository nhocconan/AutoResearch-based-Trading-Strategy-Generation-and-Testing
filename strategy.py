#!/usr/bin/env python3
"""
6h_trix_1d_volume_regime_v1
Hypothesis: On 6-hour timeframe, use TRIX (triple-smoothed EMA) from daily timeframe for trend direction, combined with volume confirmation on 6h.
Enter long when daily TRIX turns positive AND 6h volume > 1.5x 20-period average.
Enter short when daily TRIX turns negative AND 6h volume > 1.5x 20-period average.
Exit when TRIX reverses sign or volume drops below average.
TRIX filters noise and catches sustained momentum; volume confirms institutional participation.
Daily timeframe filter ensures alignment with higher timeframe trend, reducing whipsaw in choppy markets.
Target: 15-25 trades/year to minimize fee drag while capturing sustained moves.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_trix_1d_volume_regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price and volume data
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for TRIX calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    d_close = df_1d['close'].values
    
    # Calculate TRIX on daily: triple EMA of % change
    # TRIX = EMA(EMA(EMA(close, period), period), period) - 1
    period = 15
    ema1 = pd.Series(d_close).ewm(span=period, adjust=False).mean()
    ema2 = pd.Series(ema1).ewm(span=period, adjust=False).mean()
    ema3 = pd.Series(ema2).ewm(span=period, adjust=False).mean()
    
    # Calculate % change of triple EMA
    trix_raw = ema3.pct_change() * 100  # as percentage
    trix = trix_raw.values
    
    # Align TRIX to 6h timeframe
    trix_aligned = align_htf_to_ltf(prices, df_1d, trix)
    
    # Volume filter: 6h volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean()
    vol_ratio = vol_series / vol_ma
    vol_ratio = vol_ratio.fillna(0).values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(1, n):  # Start from 1 for pct_change
        # Skip if TRIX not available
        if np.isnan(trix_aligned[i]):
            signals[i] = 0.0
            continue
        
        # TRIX direction
        trix_pos = trix_aligned[i] > 0
        trix_neg = trix_aligned[i] < 0
        
        # Volume confirmation
        vol_confirmed = vol_ratio[i] > 1.5
        
        if position == 1:  # Long position
            # Exit conditions
            exit_long = False
            # Exit when TRIX turns negative
            if trix_neg:
                exit_long = True
            # Exit when volume drops below average
            elif vol_ratio[i] < 1.0:
                exit_long = True
            
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions
            exit_short = False
            # Exit when TRIX turns positive
            if trix_pos:
                exit_short = True
            # Exit when volume drops below average
            elif vol_ratio[i] < 1.0:
                exit_short = True
            
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: TRIX positive AND volume confirmed
            long_entry = trix_pos and vol_confirmed
            
            # Short entry: TRIX negative AND volume confirmed
            short_entry = trix_neg and vol_confirmed
            
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
    
    return signals