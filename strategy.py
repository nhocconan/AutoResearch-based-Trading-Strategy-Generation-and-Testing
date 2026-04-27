# #!/usr/bin/env python3
"""
Hypothesis: 4-hour Bollinger Band breakout with volume confirmation and 12-hour trend filter.
Trades only during high-volume breakouts in the direction of the 12-hour trend.
Designed to work in both bull and bear markets by using the 12-hour trend as filter.
Target: 20-50 trades/year per symbol (80-200 total over 4 years) to minimize fee drag.
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
    
    # Get 12-hour data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12-hour EMA(50) for trend
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Get 4-hour data for Bollinger Bands and volume filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4-hour Bollinger Bands (20, 2)
    close_4h = df_4h['close'].values
    ma_20 = pd.Series(close_4h).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_4h).rolling(window=20, min_periods=20).std().values
    upper = ma_20 + 2 * std_20
    lower = ma_20 - 2 * std_20
    
    # Calculate 4-hour volume MA(20)
    vol_4h = df_4h['volume'].values
    vol_ma_20_4h = pd.Series(vol_4h).rolling(window=20, min_periods=20).mean().values
    
    # Align all 4h indicators
    upper_aligned = align_htf_to_ltf(prices, df_4h, upper)
    lower_aligned = align_htf_to_ltf(prices, df_4h, lower)
    ma_20_aligned = align_htf_to_ltf(prices, df_4h, ma_20)
    vol_ma_20_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_20_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # Warmup: need Bollinger Bands, volume MA, and 12h EMA
    start_idx = max(20, 20, 50)  # max of lookbacks
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(ma_20_aligned[i]) or np.isnan(vol_ma_20_4h_aligned[i]) or 
            np.isnan(ema_50_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        upper_band = upper_aligned[i]
        lower_band = lower_aligned[i]
        ma = ma_20_aligned[i]
        vol_now = volume[i]
        vol_ma = vol_ma_20_4h_aligned[i]
        trend_12h = ema_50_12h_aligned[i]
        
        # Volume filter: volume > 1.5x 4h average (moderate to balance trades)
        vol_filter = vol_now > 1.5 * vol_ma
        
        # Entry conditions: Bollinger Band breakout with volume and 12h trend alignment
        if position == 0:
            # Long: break above upper band + volume + 12h uptrend
            if close[i] > upper_band and vol_filter and close[i] > trend_12h:
                signals[i] = size
                position = 1
            # Short: break below lower band + volume + 12h downtrend
            elif close[i] < lower_band and vol_filter and close[i] < trend_12h:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: close below 12h EMA or middle band
            if close[i] < trend_12h or close[i] < ma:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: close above 12h EMA or middle band
            if close[i] > trend_12h or close[i] > ma:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_BollingerBreakout_Volume_12hTrendFilter"
timeframe = "4h"
leverage = 1.0