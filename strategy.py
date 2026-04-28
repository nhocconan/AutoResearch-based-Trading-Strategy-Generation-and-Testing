# 12h_CamarillaPivot_R1S1_Breakout_1dTrend_Volume
# Hypothesis: Use 12-hour Camarilla R1/S1 breakouts aligned with daily trend (via 21 EMA) and volume confirmation. The daily trend filter avoids counter-trend trades, while Camarilla pivot levels provide high-probability breakout zones. Volume surge confirms institutional participation. Designed for low trade frequency (~12-37/year) to minimize fee drag and maximize robustness in both bull and bear markets.

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
    
    # Get daily data for trend filter and pivot calculation
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 21:
        return np.zeros(n)
    
    # Calculate daily 21 EMA for trend filter
    close_daily = df_daily['close'].values
    ema21_daily = pd.Series(close_daily).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Align daily EMA to 12h timeframe
    ema21_daily_aligned = align_htf_to_ltf(prices, df_daily, ema21_daily)
    
    # Daily trend: bullish when close > EMA21
    daily_uptrend = close_daily > ema21_daily
    daily_uptrend_aligned = align_htf_to_ltf(prices, df_daily, daily_uptrend.astype(float)) > 0.5
    
    # Calculate Camarilla pivot levels from previous day
    # Using high, low, close from previous daily bar
    prev_high = df_daily['high'].shift(1).values
    prev_low = df_daily['low'].shift(1).values
    prev_close = df_daily['close'].shift(1).values
    
    # Calculate pivot point
    pivot = (prev_high + prev_low + prev_close) / 3.0
    
    # Calculate Camarilla levels
    # R1 = close + (high - low) * 1.1/12
    # S1 = close - (high - low) * 1.1/12
    rang = prev_high - prev_low
    r1 = prev_close + rang * 1.1 / 12.0
    s1 = prev_close - rang * 1.1 / 12.0
    
    # Align Camarilla levels to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_daily, r1)
    s1_aligned = align_htf_to_ltf(prices, df_daily, s1)
    
    # Volume confirmation: current volume > 1.5x 50-period average
    vol_ma_50 = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    volume_surge = volume > (vol_ma_50 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema21_daily_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(volume_surge[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions with daily trend alignment and volume surge
        long_entry = close[i] > r1_aligned[i] and daily_uptrend_aligned[i] and volume_surge[i]
        short_entry = close[i] < s1_aligned[i] and not daily_uptrend_aligned[i] and volume_surge[i]
        
        # Exit on opposite Camarilla level touch with volume surge
        long_exit = close[i] < s1_aligned[i] and volume_surge[i]
        short_exit = close[i] > r1_aligned[i] and volume_surge[i]
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = -0.25  # Reverse to short
            position = -1
        elif short_exit and position == -1:
            signals[i] = 0.25   # Reverse to long
            position = 1
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_CamarillaPivot_R1S1_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0