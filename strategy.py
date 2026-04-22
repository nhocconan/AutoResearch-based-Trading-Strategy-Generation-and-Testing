#!/usr/bin/env python3
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
    
    # Load 1d data for ATR and EMA (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d ATR(14) for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]))
    tr2 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.inf], np.maximum(tr1, tr2)])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 1d EMA(34) for trend filter
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align indicators to 4h timeframe
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # 4h Donchian(20) breakout levels
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(atr_14_aligned[i]) or np.isnan(ema_34_aligned[i]) or 
            np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or 
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above Donchian high + ATR filter + Uptrend
            if (close[i] > highest_20[i] and 
                atr_14_aligned[i] > 0 and  # volatility present
                close[i] > ema_34_aligned[i] and  # uptrend filter
                volume[i] > 1.5 * vol_avg_20[i]):  # volume confirmation
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low + ATR filter + Downtrend
            elif (close[i] < lowest_20[i] and 
                  atr_14_aligned[i] > 0 and  # volatility present
                  close[i] < ema_34_aligned[i] and  # downtrend filter
                  volume[i] > 1.5 * vol_avg_20[i]):  # volume confirmation
                signals[i] = -0.25
                position = -1
        else:
            # ATR-based trailing stop
            if position == 1:
                # Long position: trail from highest high since entry
                if i > 0:
                    # Update highest high since entry (simplified: use rolling max)
                    highest_since_entry = np.maximum.accumulate(high[max(0, i-50):i+1])[-1] if i > 0 else high[i]
                    if close[i] < highest_since_entry - 2.0 * atr_14_aligned[i]:
                        signals[i] = 0.0
                        position = 0
                    else:
                        signals[i] = 0.25
            else:  # position == -1
                # Short position: trail from lowest low since entry
                if i > 0:
                    lowest_since_entry = np.minimum.accumulate(low[max(0, i-50):i+1])[-1] if i > 0 else low[i]
                    if close[i] > lowest_since_entry + 2.0 * atr_14_aligned[i]:
                        signals[i] = 0.0
                        position = 0
                    else:
                        signals[i] = -0.25
    
    return signals

name = "4H_Donchian20_ATR_EMA_Volume_Breakout"
timeframe = "4h"
leverage = 1.0