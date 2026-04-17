#!/usr/bin/env python3
"""
4h Williams %R Mean Reversion with Volume Spike and Trend Filter
Long: Williams %R < -80 + volume > 1.5x 4h volume MA(20) + price > 4h EMA50
Short: Williams %R > -20 + volume > 1.5x 4h volume MA(20) + price < 4h EMA50
Exit: Williams %R crosses above -50 (long) or below -50 (short)
Williams %R calculated on 14-period lookback
Target: 20-30 trades/year per symbol
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
    
    # Calculate Williams %R on 4h data
    df_4h = get_htf_data(prices, '4h')
    highest_high = pd.Series(df_4h['high']).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(df_4h['low']).rolling(window=14, min_periods=14).min()
    williams_r = -100 * (highest_high - df_4h['close']) / (highest_high - lowest_low)
    williams_r = williams_r.replace(0, np.nan).values  # avoid division by zero
    
    # 4h EMA50 for trend filter
    ema_50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # 4h volume MA(20) for volume confirmation
    volume_ma_20 = pd.Series(df_4h['volume']).rolling(window=20, min_periods=20).mean().values
    
    # Align all 4h indicators to lower timeframe (assuming 4h is primary)
    williams_r_aligned = align_htf_to_ltf(prices, df_4h, williams_r)
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    volume_ma_20_aligned = align_htf_to_ltf(prices, df_4h, volume_ma_20)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    entry_price = 0.0
    
    start_idx = 50  # warmup
    
    for i in range(start_idx, n):
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(volume_ma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = volume_ma_20_aligned[i]
        wr = williams_r_aligned[i]
        
        if position == 0:
            # Long: oversold + volume + uptrend
            if wr < -80 and vol > 1.5 * vol_ma and price > ema_50_4h_aligned[i]:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: overbought + volume + downtrend
            elif wr > -20 and vol > 1.5 * vol_ma and price < ema_50_4h_aligned[i]:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: Williams %R crosses above -50
            if wr > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Williams %R crosses below -50
            if wr < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WilliamsR_MeanReversion_Volume_Trend"
timeframe = "4h"
leverage = 1.0