#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Keltner Channel breakout with volume confirmation and weekly trend filter
# Long when price breaks above Keltner upper band with volume >1.5x 20-period average and weekly close above 20-period EMA
# Short when price breaks below Keltner lower band with volume >1.5x 20-period average and weekly close below 20-period EMA
# Exit when price crosses the 6h EMA(20)
# Weekly trend filter prevents counter-trend trades, reducing false breakouts
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 6h and weekly data ONCE before loop
    df_6h = get_htf_data(prices, '6h')
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate 6h EMA(20) for ATR basis and exit signal
    close_6h = df_6h['close'].values
    ema_20_6h = pd.Series(close_6h).ewm(span=20, min_periods=20, adjust=False).mean().values
    
    # Calculate ATR(10) for Keltner Channel
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    tr1 = high_6h - low_6h
    tr2 = np.abs(high_6h - np.roll(close_6h, 1))
    tr3 = np.abs(low_6h - np.roll(close_6h, 1))
    tr2[0] = tr1[0]  # First value
    tr3[0] = tr1[0]  # First value
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_10 = pd.Series(tr).ewm(span=10, min_periods=10, adjust=False).mean().values
    
    # Calculate Keltner Channel (20, 10, 2.0)
    keltner_upper = ema_20_6h + 2.0 * atr_10
    keltner_lower = ema_20_6h - 2.0 * atr_10
    
    # Calculate 6h volume average (20-period)
    vol_6h = df_6h['volume'].values
    vol_ma_6h = pd.Series(vol_6h).rolling(window=20, min_periods=20).mean().values
    
    # Calculate weekly EMA(20) for trend filter
    close_weekly = df_weekly['close'].values
    ema_20_weekly = pd.Series(close_weekly).ewm(span=20, min_periods=20, adjust=False).mean().values
    
    # Align indicators to 6h timeframe
    keltner_upper_aligned = align_htf_to_ltf(prices, df_6h, keltner_upper)
    keltner_lower_aligned = align_htf_to_ltf(prices, df_6h, keltner_lower)
    ema_20_6h_aligned = align_htf_to_ltf(prices, df_6h, ema_20_6h)
    vol_ma_6h_aligned = align_htf_to_ltf(prices, df_6h, vol_ma_6h)
    ema_20_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_20_weekly)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 30  # for 20-period EMA and ATR calculations
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(keltner_upper_aligned[i]) or np.isnan(keltner_lower_aligned[i]) or 
            np.isnan(ema_20_6h_aligned[i]) or np.isnan(vol_ma_6h_aligned[i]) or 
            np.isnan(ema_20_weekly_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_6h_current = volume[i]  # Current 6h volume
        
        if position == 0:
            # Long setup: break above Keltner upper with volume confirmation and weekly uptrend
            if (price > keltner_upper_aligned[i] and 
                vol_6h_current > 1.5 * vol_ma_6h_aligned[i] and  # Volume confirmation
                price > ema_20_weekly_aligned[i]):              # Weekly uptrend filter
                position = 1
                signals[i] = position_size
            # Short setup: break below Keltner lower with volume confirmation and weekly downtrend
            elif (price < keltner_lower_aligned[i] and 
                  vol_6h_current > 1.5 * vol_ma_6h_aligned[i] and  # Volume confirmation
                  price < ema_20_weekly_aligned[i]):              # Weekly downtrend filter
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below 6h EMA(20)
            if price < ema_20_6h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses above 6h EMA(20)
            if price > ema_20_6h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_Keltner_WeeklyTrend_Volume"
timeframe = "6h"
leverage = 1.0