#!/usr/bin/env python3
name = "12h_1W_Keltner_Breakout_Trend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1W data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # 1W EMA20 for trend filter
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # 1W ATR(10) for Keltner Channel
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr_1w = np.concatenate([[np.max([high_1w[0] - low_1w[0], np.abs(high_1w[0] - close_1w[0]), np.abs(low_1w[0] - close_1w[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_1w = pd.Series(tr_1w).ewm(alpha=1/10, adjust=False, min_periods=10).mean().values
    
    # 1W Keltner Channel (20, 10, 2.0)
    upper_keltner = ema_20_1w + (2.0 * atr_1w)
    lower_keltner = ema_20_1w - (2.0 * atr_1w)
    upper_keltner_aligned = align_htf_to_ltf(prices, df_1w, upper_keltner)
    lower_keltner_aligned = align_htf_to_ltf(prices, df_1w, lower_keltner)
    
    # 12h ATR(10) for volatility filter
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(alpha=1/10, adjust=False, min_periods=10).mean().values
    atr_ma = pd.Series(atr).rolling(window=10, min_periods=10).mean().values
    atr_ratio = atr / atr_ma
    atr_ratio = np.nan_to_num(atr_ratio, nan=1.0)
    
    # 12h Volume ratio (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_keltner_aligned[i]) or np.isnan(lower_keltner_aligned[i]) or 
            np.isnan(ema_20_1w_aligned[i]) or np.isnan(atr_ratio[i]) or 
            np.isnan(vol_ratio[i])):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volatility filter: avoid low volatility periods
        vol_filter = atr_ratio[i] > 0.8
        
        # Volume threshold
        volume_surge = vol_ratio[i] > 1.5
        
        if position == 0:
            # Long: Price breaks above upper Keltner with volume surge and above 1W EMA20
            if (close[i] > upper_keltner_aligned[i] and 
                volume_surge and 
                vol_filter and 
                close[i] > ema_20_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below lower Keltner with volume surge and below 1W EMA20
            elif (close[i] < lower_keltner_aligned[i] and 
                  volume_surge and 
                  vol_filter and 
                  close[i] < ema_20_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit long: Price returns below EMA20 or volatility drops
                if (close[i] < ema_20_1w_aligned[i]) or (atr_ratio[i] < 0.6):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: Price returns above EMA20 or volatility drops
                if (close[i] > ema_20_1w_aligned[i]) or (atr_ratio[i] < 0.6):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals