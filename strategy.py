#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_Donchian_Breakout_Volume_Trend_Filter_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data once
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly EMA(34) for trend direction
    close_1w = df_1w['close'].values
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Calculate daily Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike detection: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Calculate daily range for volatility filter (ATR-like)
    true_range = np.maximum(high - low, 
                           np.maximum(np.abs(high - np.roll(close, 1)), 
                                    np.abs(low - np.roll(close, 1))))
    atr_10 = pd.Series(true_range).rolling(window=10, min_periods=10).mean().values
    atr_pct = atr_10 / close  # ATR as percentage of price
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 35  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema34_1w_aligned[i]) or np.isnan(high_20[i]) or 
            np.isnan(low_20[i]) or np.isnan(atr_pct[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_val = ema34_1w_aligned[i]
        upper_channel = high_20[i]
        lower_channel = low_20[i]
        vol_spike = volume_spike[i]
        volatility = atr_pct[i]
        
        # Volatility filter: only trade when volatility is between 1% and 5%
        vol_filter = (volatility >= 0.01) and (volatility <= 0.05)
        
        if position == 0:
            # Enter long: price breaks above upper channel with volume spike, above weekly EMA, in volatility range
            if (close[i] > upper_channel and vol_spike and 
                close[i] > ema_val and vol_filter):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below lower channel with volume spike, below weekly EMA, in volatility range
            elif (close[i] < lower_channel and vol_spike and 
                  close[i] < ema_val and vol_filter):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below lower channel OR below weekly EMA
            if (close[i] < lower_channel or close[i] < ema_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above upper channel OR above weekly EMA
            if (close[i] > upper_channel or close[i] > ema_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals