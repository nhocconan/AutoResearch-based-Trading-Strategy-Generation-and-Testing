#!/usr/bin/env python3
"""
6h_ImpulseMACD_WeeklyTrend_VolumeFilter
Hypothesis: 6h MACD histogram impulse (rising/falling) aligned with weekly EMA50 trend and volume confirmation.
Long when MACD histogram > 0 and rising (bullish impulse), weekly uptrend, volume > 1.5x average.
Short when MACD histogram < 0 and falling (bearish impulse), weekly downtrend, volume > 1.5x average.
Exit when MACD histogram crosses zero or price violates weekly trend.
Designed for moderate trade frequency (target: 20-40 trades/year) to balance signal quality and fees.
Works in bull/bear via weekly trend alignment and MACD impulse as momentum filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (weekly for trend)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # === Weekly EMA50 for HTF trend filter ===
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # === 6h MACD Histogram (12,26,9) ===
    close = prices['close'].values
    ema12 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema26 = pd.Series(close).ewm(span=26, adjust=False, min_periods=26).mean().values
    macd = ema12 - ema26
    signal_line = pd.Series(macd).ewm(span=9, adjust=False, min_periods=9).mean().values
    macd_hist = macd - signal_line
    
    # === Volume confirmation (20-period average) ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(60, n):
        # Skip if indicators not ready
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(macd_hist[i]) 
            or np.isnan(signal_line[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Volume confirmation: current volume > 1.5x 20-period average
            vol_confirm = volume[i] > 1.5 * vol_ma[i]
            
            # Long conditions: MACD hist > 0 and rising (bullish impulse), weekly uptrend, volume spike
            macd_bullish = macd_hist[i] > 0 and macd_hist[i] > macd_hist[i-1]
            weekly_uptrend = price > ema_50_1w_aligned[i]
            
            # Short conditions: MACD hist < 0 and falling (bearish impulse), weekly downtrend, volume spike
            macd_bearish = macd_hist[i] < 0 and macd_hist[i] < macd_hist[i-1]
            weekly_downtrend = price < ema_50_1w_aligned[i]
            
            # Entry logic
            if macd_bullish and weekly_uptrend and vol_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = price
            elif macd_bearish and weekly_downtrend and vol_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Exit conditions: MACD hist crosses zero OR price breaks weekly trend
            macd_exit = macd_hist[i] <= 0
            trend_exit = price < ema_50_1w_aligned[i]
            
            if macd_exit or trend_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit conditions: MACD hist crosses zero OR price breaks weekly trend
            macd_exit = macd_hist[i] >= 0
            trend_exit = price > ema_50_1w_aligned[i]
            
            if macd_exit or trend_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ImpulseMACD_WeeklyTrend_VolumeFilter"
timeframe = "6h"
leverage = 1.0