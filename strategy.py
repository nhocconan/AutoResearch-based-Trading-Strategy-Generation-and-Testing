#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian channel breakout with 1-day ATR-based volatility filter and volume confirmation.
In bull markets, breakouts capture trends; in bear markets, volatility filter avoids false breakouts during low-volatility chop.
Volume confirms institutional participation. Target: 20-40 trades/year per symbol.
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
    
    # Donchian channels (20-period) on 4h data
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 1-day ATR (14-period) for volatility filter
    df_1d = get_htf_data(prices, '1d')
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = np.abs(df_1d['low'] - df_1d['close'].shift(1))
    tr_1d = np.maximum(np.maximum(tr1, tr2), tr3)
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Volume confirmation: 20-period volume MA
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 40  # warmup for Donchian and ATR
    
    for i in range(start_idx, n):
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(atr_1d_aligned[i]) or np.isnan(volume_ma_20.iloc[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = volume_ma_20.iloc[i]
        atr_val = atr_1d_aligned[i]
        
        # Volatility filter: only trade when ATR is above its 50-period median (avoid low-vol chop)
        if i >= 50:
            atr_median = np.nanmedian(atr_1d_aligned[max(0, i-49):i+1])
            vol_filter = atr_val > atr_median
        else:
            vol_filter = True  # insufficient data for median, allow trading
        
        if position == 0:
            # Long breakout: price breaks above 20-period high with volume and vol filter
            if price > highest_high[i] and vol > 1.5 * vol_ma and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short breakdown: price breaks below 20-period low with volume and vol filter
            elif price < lowest_low[i] and vol > 1.5 * vol_ma and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit when price breaks below 20-period low or volatility drops
            if price < lowest_low[i] or (i >= 50 and atr_val < np.nanmedian(atr_1d_aligned[max(0, i-49):i+1])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit when price breaks above 20-period high or volatility drops
            if price > highest_high[i] or (i >= 50 and atr_val < np.nanmedian(atr_1d_aligned[max(0, i-49):i+1])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_DonchianBreakout_Volume_VolatilityFilter"
timeframe = "4h"
leverage = 1.0