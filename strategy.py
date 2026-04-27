#!/usr/bin/env python3
"""
Hypothesis: Daily Bollinger Band squeeze breakout with weekly trend filter and volume confirmation.
Trades only during low volatility (BB squeeze) followed by expansion breakout in the direction of the weekly trend.
Designed to capture volatility breakouts in both bull and bear markets by using the weekly trend as filter.
Target: 10-25 trades/year per symbol (40-100 total over 4 years) to minimize fee drag.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA(34) for trend
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate daily Bollinger Bands (20, 2)
    close_d = pd.Series(close)
    ma_20 = close_d.rolling(window=20, min_periods=20).mean().values
    std_20 = close_d.rolling(window=20, min_periods=20).std().values
    upper = ma_20 + 2 * std_20
    lower = ma_20 - 2 * std_20
    
    # Calculate daily Bollinger Band width for squeeze detection
    bb_width = (upper - lower) / ma_20
    bb_width_ma_50 = pd.Series(bb_width).rolling(window=50, min_periods=50).mean().values
    
    # Calculate daily volume MA(20)
    volume_d = pd.Series(volume)
    vol_ma_20 = volume_d.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need BB width MA, volume MA, and weekly EMA
    start_idx = max(50, 20, 34)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ma_20[i]) or np.isnan(std_20[i]) or 
            np.isnan(bb_width_ma_50[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(ema_34_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        upper_band = upper[i]
        lower_band = lower[i]
        ma = ma_20[i]
        bb_width_now = bb_width[i]
        bb_width_ma = bb_width_ma_50[i]
        vol_now = volume[i]
        vol_ma = vol_ma_20[i]
        trend_1w = ema_34_1w_aligned[i]
        
        # Squeeze condition: BB width below 50-day average (low volatility)
        squeeze = bb_width_now < bb_width_ma
        
        # Volume filter: volume > 1.5x daily average
        vol_filter = vol_now > 1.5 * vol_ma
        
        # Entry conditions: Bollinger Band breakout after squeeze with volume and weekly trend alignment
        if position == 0:
            # Long: break above upper band after squeeze + volume + weekly uptrend
            if close[i] > upper_band and squeeze and vol_filter and close[i] > trend_1w:
                signals[i] = size
                position = 1
            # Short: break below lower band after squeeze + volume + weekly downtrend
            elif close[i] < lower_band and squeeze and vol_filter and close[i] < trend_1w:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: close below weekly EMA or Bollinger mid-band
            if close[i] < trend_1w or close[i] < ma:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: close above weekly EMA or Bollinger mid-band
            if close[i] > trend_1w or close[i] > ma:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_BollingerSqueezeBreakout_WeeklyTrendFilter_Volume"
timeframe = "1d"
leverage = 1.0