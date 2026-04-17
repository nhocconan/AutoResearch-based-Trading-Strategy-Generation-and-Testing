#!/usr/bin/env python3
"""
1h Fractal Breakout with 4h Trend and Volume Confirmation
Long: Bullish fractal break above resistance + 4h EMA50 up + volume > 1.5x 4h volume MA
Short: Bearish fractal break below support + 4h EMA50 down + volume > 1.5x 4h volume MA
Exit: Opposite fractal break
Target: 20-40 trades/year per symbol via tight fractal + volume + trend filters
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h EMA50 for trend filter
    df_4h = get_htf_data(prices, '4h')
    ema_50 = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h = align_htf_to_ltf(prices, df_4h, ema_50)
    
    # 4h volume moving average (20-period for confirmation)
    volume_ma_20 = pd.Series(df_4h['volume']).rolling(window=20, min_periods=20).mean().values
    volume_ma_20_4h = align_htf_to_ltf(prices, df_4h, volume_ma_20)
    
    # Daily fractals for support/resistance levels
    df_1d = get_htf_data(prices, '1d')
    bearish, bullish = compute_williams_fractals(df_1d['high'].values, df_1d['low'].values)
    # Require 2 additional bars for fractal confirmation (standard)
    bearish_1d = align_htf_to_ltf(prices, df_1d, bearish, additional_delay_bars=2)
    bullish_1d = align_htf_to_ltf(prices, df_1d, bullish, additional_delay_bars=2)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 100  # warmup
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_4h[i]) or np.isnan(volume_ma_20_4h[i]) or 
            np.isnan(bearish_1d[i]) or np.isnan(bullish_1d[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = volume_ma_20_4h[i]
        
        if position == 0:
            # Long: bullish fractal break above resistance + 4h EMA up + volume
            if (bullish_1d[i] > 0 and  # bullish fractal present
                price > bullish_1d[i] and 
                ema_50_4h[i] > ema_50_4h[i-1] and  # 4h EMA50 rising
                vol > 1.5 * vol_ma):
                signals[i] = 0.20
                position = 1
            # Short: bearish fractal break below support + 4h EMA down + volume
            elif (bearish_1d[i] > 0 and  # bearish fractal present
                  price < bearish_1d[i] and 
                  ema_50_4h[i] < ema_50_4h[i-1] and  # 4h EMA50 falling
                  vol > 1.5 * vol_ma):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: bearish fractal break below support
            if bearish_1d[i] > 0 and price < bearish_1d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: bullish fractal break above resistance
            if bullish_1d[i] > 0 and price > bullish_1d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Fractal_Breakout_4hTrend_Volume"
timeframe = "1h"
leverage = 1.0