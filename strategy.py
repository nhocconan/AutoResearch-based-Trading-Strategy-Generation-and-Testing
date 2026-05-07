#!/usr/bin/env python3
name = "6h_ElderRay_Momentum_1dTrend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Elder Ray (Bull/Bear Power) on 6h data
    ema13 = np.full(n, np.nan)
    if n >= 13:
        close_s = pd.Series(close)
        ema13 = close_s.ewm(span=13, adjust=False, min_periods=13).values
    
    bull_power = high - ema13
    bear_power = low - ema13
    
    # 1d trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 21:
        return np.zeros(n)
    
    ema21_1d = pd.Series(df_1d['close'].values).ewm(span=21, adjust=False, min_periods=21).values
    ema21_1d_aligned = align_htf_to_ltf(prices, df_1d, ema21_1d)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    vol_filter = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 3  # ~18 hours
    
    start_idx = max(13, 20)  # EMA13 and vol_ma_20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema13[i]) or 
            np.isnan(ema21_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                bars_since_last_trade += 1
            continue
        
        bars_since_last_trade += 1
        
        # 1d trend: price above/below EMA21
        trend_up = close[i] > ema21_1d_aligned[i]
        trend_down = close[i] < ema21_1d_aligned[i]
        
        if position == 0 and bars_since_last_trade >= cooldown_bars:
            # Long: Bull Power > 0 and rising, in 1d uptrend, with volume
            if (bull_power[i] > 0 and 
                bull_power[i] > bull_power[i-1] and 
                trend_up and 
                vol_filter[i]):
                signals[i] = 0.25
                position = 1
                bars_since_last_trade = 0
            # Short: Bear Power < 0 and falling, in 1d downtrend, with volume
            elif (bear_power[i] < 0 and 
                  bear_power[i] < bear_power[i-1] and 
                  trend_down and 
                  vol_filter[i]):
                signals[i] = -0.25
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit: Bull Power <= 0 or 1d trend changes to down
            if bull_power[i] <= 0 or not trend_up:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Bear Power >= 0 or 1d trend changes to up
            if bear_power[i] >= 0 or not trend_down:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Elder Ray (Bull/Bear Power) measures buying/selling pressure relative to EMA13. 
# Combining with 1d EMA21 trend filter and volume confirmation captures momentum with institutional validation. 
# Works in bull markets (bull power rising in uptrend) and bear markets (bear power falling in downtrend). 
# Target: 50-150 trades over 4 years (12-37/year) to minimize fee drag while capturing sustained moves. 
# Elder Ray is less common than RSI/MACD, offering a fresh edge on 6s timeframe.