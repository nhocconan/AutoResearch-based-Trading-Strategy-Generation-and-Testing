#!/usr/bin/env python3
"""
12h Camarilla Pivot (R1/S1) Breakout + 1d Volume Spike + 1d EMA(50) Trend Filter
Long: Price breaks above R1 with volume > 1.5x 20-period 1d MA and price > EMA50
Short: Price breaks below S1 with volume > 1.5x 20-period 1d MA and price < EMA50
Exit: Opposite Camarilla level touch (S1 for long, R1 for short)
Position size: 0.25
Target: 15-30 trades/year (60-120 total over 4 years) to avoid fee drag
Works in bull/bear via trend filter + volume confirmation on breakouts
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
    
    # 1d Camarilla pivot levels (based on previous day)
    df_1d = get_htf_data(prices, '1d')
    # Calculate Camarilla levels: based on previous day's OHLC
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    prev_close = df_1d['close'].shift(1)
    prev_high = df_1d['high'].shift(1)
    prev_low = df_1d['low'].shift(1)
    camarilla_r1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    camarilla_s1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    camarilla_r1_1d = align_htf_to_ltf(prices, df_1d, camarilla_r1.values)
    camarilla_s1_1d = align_htf_to_ltf(prices, df_1d, camarilla_s1.values)
    
    # 1d EMA50 for trend filter
    ema_50 = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # 1d volume spike filter (1.5x 20-period MA)
    volume_ma_20 = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean()
    volume_ma_20_1d = align_htf_to_ltf(prices, df_1d, volume_ma_20.values)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    entry_price = 0.0
    
    start_idx = 50  # warmup
    
    for i in range(start_idx, n):
        if (np.isnan(camarilla_r1_1d[i]) or np.isnan(camarilla_s1_1d[i]) or 
            np.isnan(ema_50_1d[i]) or np.isnan(volume_ma_20_1d[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = volume_ma_20_1d[i]
        
        if position == 0:
            # Long: break above R1 + trend filter + volume spike
            if price > camarilla_r1_1d[i] and price > ema_50_1d[i] and vol > 1.5 * vol_ma:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: break below S1 + trend filter + volume spike
            elif price < camarilla_s1_1d[i] and price < ema_50_1d[i] and vol > 1.5 * vol_ma:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: price touches S1 (opposite level)
            if price < camarilla_s1_1d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price touches R1 (opposite level)
            if price > camarilla_r1_1d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1S1_1dEMA50_VolumeSpike"
timeframe = "12h"
leverage = 1.0