#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_Trix_Volume_Trend_Filter"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 15:
        return np.zeros(n)
    
    # === 1d: TRIX (15-period) ===
    close_1d = df_1d['close'].values
    # TRIX = EMA(EMA(EMA(close, 15), 15), 15) - 1, then * 100 for percentage
    ema1 = pd.Series(close_1d).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    trix = (ema3[1:] - ema3[:-1]) / ema3[:-1] * 100
    # Prepend first value to match length
    trix = np.concatenate([[0], trix])
    trix_1d_aligned = align_htf_to_ltf(prices, df_1d, trix)
    
    # === 4h: Price and volume ===
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume ratio (current vs 20-period average)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip outside session
        if not (8 <= hours[i] <= 20):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Get values
        close_val = close[i]
        vol_ratio_val = vol_ratio[i]
        trix_val = trix_1d_aligned[i]
        
        # Skip if any value is NaN
        if np.isnan(vol_ratio_val) or np.isnan(trix_val):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Positive TRIX + volume confirmation
            if (trix_val > 0 and          # TRIX positive (bullish momentum)
                vol_ratio_val > 1.3):     # Volume confirmation
                signals[i] = 0.25
                position = 1
            # Short: Negative TRIX + volume confirmation
            elif (trix_val < 0 and        # TRIX negative (bearish momentum)
                  vol_ratio_val > 1.3):   # Volume confirmation
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: TRIX turns negative or volume drops
            if (trix_val < 0 or           # TRIX turned negative
                vol_ratio_val < 0.7):     # Low volume (losing momentum)
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: TRIX turns positive or volume drops
            if (trix_val > 0 or           # TRIX turned positive
                vol_ratio_val < 0.7):     # Low volume (losing momentum)
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals