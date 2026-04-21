#!/usr/bin/env python3
"""
12h_1d_KAMA_Trend_With_Volume_Confirmation
Hypothesis: KAMA adapts to market noise - in trending markets it follows price closely, in ranging markets it stays flat. 
We go long when price is above KAMA with volume confirmation, short when below KAMA with volume confirmation.
Uses 1d trend filter to avoid counter-trend trades. Designed for low trade frequency (12-37/year) to minimize fee drag.
Works in both bull and bear markets by adapting to market conditions via adaptive moving average.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data for trend filter
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 20:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_daily = df_daily['close'].values
    ema_34_daily = pd.Series(close_daily).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_daily_aligned = align_htf_to_ltf(prices, df_daily, ema_34_daily)
    
    # Main timeframe data (12h)
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate KAMA (Kaufman Adaptive Moving Average)
    # ER = |Close - Close[10]| / Sum(|Close - Close[1]|) over 10 periods
    change = np.abs(np.diff(close, n=10))  # |Close - Close[10]|
    volatility = np.zeros_like(close)
    for i in range(10, len(close)):
        volatility[i] = np.sum(np.abs(np.diff(close[i-9:i+1], n=1)))  # Sum of 10 absolute changes
    
    er = np.zeros_like(close)
    er[10:] = change[10:] / np.where(volatility[10:] != 0, volatility[10:], 1)
    
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # Fast SC=2/(2+1), Slow SC=2/(30+1)
    
    kama = np.full_like(close, np.nan)
    kama[0] = close[0]
    for i in range(1, len(close)):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # Volume filter: current volume > 1.3x 20-period average
    volume_avg = np.zeros_like(volume)
    for i in range(len(volume)):
        if i >= 20:
            volume_avg[i] = np.mean(volume[i-20:i])
        else:
            volume_avg[i] = np.mean(volume[:i+1]) if i > 0 else volume[i]
    volume_filter = volume > (1.3 * volume_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(34, n):
        # Skip if NaN in critical values
        if (np.isnan(kama[i]) or np.isnan(ema_34_daily_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        kama_val = kama[i]
        ema_34 = ema_34_daily_aligned[i]
        vol_ok = volume_filter[i]
        
        if position == 0:
            # Long: price above KAMA and above daily EMA34 (uptrend) with volume
            if price > kama_val and price > ema_34 and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA and below daily EMA34 (downtrend) with volume
            elif price < kama_val and price < ema_34 and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below KAMA or trend turns down
            if price < kama_val or price < ema_34:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above KAMA or trend turns up
            if price > kama_val or price > ema_34:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1d_KAMA_Trend_With_Volume_Confirmation"
timeframe = "12h"
leverage = 1.0