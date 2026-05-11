#!/usr/bin/env python3
name = "6h_ADX_DMI_Trend_With_1d_Volume_Filter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate ADX and DMI on 6h data (period=14)
    # +DM, -DM, TR
    up_move = high[1:] - high[:-1]
    down_move = low[:-1] - low[1:]
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Wilder's smoothing (alpha = 1/period)
    period = 14
    alpha = 1.0 / period
    
    def wilder_smooth(arr):
        res = np.zeros_like(arr)
        res[0] = arr[0] if len(arr) > 0 else 0
        for i in range(1, len(arr)):
            res[i] = alpha * arr[i] + (1 - alpha) * res[i-1]
        return res
    
    # Pad arrays to match original length
    plus_dm_padded = np.concatenate([[0], plus_dm])
    minus_dm_padded = np.concatenate([[0], minus_dm])
    tr_padded = np.concatenate([[0], tr])
    
    atr = wilder_smooth(tr_padded)
    plus_di = 100 * wilder_smooth(plus_dm_padded) / (atr + 1e-10)
    minus_di = 100 * wilder_smooth(minus_dm_padded) / (atr + 1e-10)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = wilder_smooth(dx)
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    vol_ma20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align indicators to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, pd.DataFrame({'index': range(len(adx))}), adx)
    plus_di_aligned = align_htf_to_ltf(prices, pd.DataFrame({'index': range(len(plus_di))}), plus_di)
    minus_di_aligned = align_htf_to_ltf(prices, pd.DataFrame({'index': range(len(minus_di))}), minus_di)
    vol_ma20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 30)  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(adx_aligned[i]) or 
            np.isnan(plus_di_aligned[i]) or
            np.isnan(minus_di_aligned[i]) or
            np.isnan(vol_ma20_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # ADX > 25 indicates strong trend
        # +DI > -DI indicates uptrend, -DI > +DI indicates downtrend
        if position == 0:
            # Long: ADX > 25, +DI > -DI, and volume confirmation
            if (adx_aligned[i] > 25 and 
                plus_di_aligned[i] > minus_di_aligned[i] and 
                volume[i] > 1.5 * vol_ma20_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: ADX > 25, -DI > +DI, and volume confirmation
            elif (adx_aligned[i] > 25 and 
                  minus_di_aligned[i] > plus_di_aligned[i] and 
                  volume[i] > 1.5 * vol_ma20_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: ADX < 20 (weak trend) or -DI > +DI (trend reversal)
            if (adx_aligned[i] < 20 or 
                minus_di_aligned[i] > plus_di_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: ADX < 20 (weak trend) or +DI > -DI (trend reversal)
            if (adx_aligned[i] < 20 or 
                plus_di_aligned[i] > minus_di_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals