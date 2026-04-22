#!/usr/bin/env python3
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
    
    # Load 1d data for trend and volatility filters (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA 34 for trend direction
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 1d ATR for volatility filter (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # 6h 20-period Donchian channels
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 6h volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(atr_1d_aligned[i]) or
            np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above 20-period high + above 1d EMA34 + volume spike + volatility filter
            if (close[i] > high_roll[i] and 
                close[i] > ema_34_1d_aligned[i] and
                volume[i] > 1.5 * vol_avg_20[i] and
                atr_1d_aligned[i] > 0.5 * np.nanmedian(atr_1d_aligned[max(0, i-20):i])):  # Volatility above median
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below 20-period low + below 1d EMA34 + volume spike + volatility filter
            elif (close[i] < low_roll[i] and 
                  close[i] < ema_34_1d_aligned[i] and
                  volume[i] > 1.5 * vol_avg_20[i] and
                  atr_1d_aligned[i] > 0.5 * np.nanmedian(atr_1d_aligned[max(0, i-20):i])):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price crosses back to 20-period opposite band
            if position == 1:
                # Exit long: Price closes below 20-period low
                if close[i] < low_roll[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short: Price closes above 20-period high
                if close[i] > high_roll[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6H_Donchian20_1dEMA34_Volume_Volatility"
timeframe = "6h"
leverage = 1.0