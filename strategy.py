#!/usr/bin/env python3
"""
4h 1D Close Reversion with Volume Spike and Trend Filter
Long: Price < 1d close AND volume > 2x 4h volume MA(20) AND price > 4h EMA34
Short: Price > 1d close AND volume > 2x 4h volume MA(20) AND price < 4h EMA34
Exit: Price crosses 1d close or opposite signal
Designed for mean reversion in ranging markets with trend filter to avoid whipsaw.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d close (mean reversion target)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close']
    close_1d_shifted = close_1d.shift(1)  # Previous day's close
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d_shifted.values)
    
    # 4h EMA34 for trend filter
    df_4h = get_htf_data(prices, '4h')
    ema_34 = pd.Series(df_4h['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_4h, ema_34)
    
    # 4h volume moving average (20-period)
    volume_ma_20 = pd.Series(df_4h['volume']).rolling(window=20, min_periods=20).mean()
    volume_ma_20_aligned = align_htf_to_ltf(prices, df_4h, volume_ma_20.values)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 40  # warmup
    
    for i in range(start_idx, n):
        if (np.isnan(close_1d_aligned[i]) or np.isnan(ema_34_aligned[i]) or 
            np.isnan(volume_ma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = volume_ma_20_aligned[i]
        target_close = close_1d_aligned[i]
        trend = ema_34_aligned[i]
        
        if position == 0:
            # Long: price below 1d close (oversold) + volume spike + above trend
            if price < target_close and vol > 2.0 * vol_ma and price > trend:
                signals[i] = 0.25
                position = 1
            # Short: price above 1d close (overbought) + volume spike + below trend
            elif price > target_close and vol > 2.0 * vol_ma and price < trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses above 1d close (mean reversion complete)
            if price >= target_close:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses below 1d close (mean reversion complete)
            if price <= target_close:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_1DClose_Reversion_VolumeSpike_TrendFilter"
timeframe = "4h"
leverage = 1.0