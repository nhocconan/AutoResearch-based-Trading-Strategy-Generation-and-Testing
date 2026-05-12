#!/usr/bin/env python3
name = "4h_BollingerBreakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

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
    
    # 1d trend filter: EMA50
    df_1d = get_htf_data(prices, '1d')
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Bollinger Bands with ATR-based deviation
    # Calculate ATR(21)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Middle band: SMA(21)
    close_pd = pd.Series(close)
    sma21 = close_pd.rolling(window=21, min_periods=21).mean().values
    
    # Upper and lower bands: SMA ± 2*ATR
    upper_band = sma21 + 2 * atr
    lower_band = sma21 - 2 * atr
    
    # Volume filter: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # need enough data for longest indicator
    
    for i in range(start_idx, n):
        # Skip if 1d trend data not ready
        if np.isnan(ema50_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Price breaks above upper band AND volume spike AND 1d uptrend
            if (close[i] > upper_band[i] and 
                volume[i] > 1.5 * vol_ma[i] and 
                close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below lower band AND volume spike AND 1d downtrend
            elif (close[i] < lower_band[i] and 
                  volume[i] > 1.5 * vol_ma[i] and 
                  close[i] < ema50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price breaks below lower band
            if close[i] < lower_band[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price breaks above upper band
            if close[i] > upper_band[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals