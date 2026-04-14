#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Bollinger Band Breakout with 12h EMA Trend Filter and Volume Spike
# Uses Bollinger Bands (20,2) for volatility-based breakout entries
# 12h EMA (50) provides multi-timeframe trend direction to avoid counter-trend trades
# Volume confirmation (>2x average) ensures institutional participation
# Designed to work in both bull and bear markets by trading breakouts in direction of higher timeframe trend
# Target: 20-50 trades/year (80-200 total over 4 years) to minimize fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h EMA data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Bollinger Bands (20, 2)
    close_series = pd.Series(close)
    ma_20 = close_series.rolling(window=20, min_periods=20).mean().values
    std_20 = close_series.rolling(window=20, min_periods=20).std().values
    upper_bb = ma_20 + 2 * std_20
    lower_bb = ma_20 - 2 * std_20
    
    # Volume confirmation: volume > 2x average volume (20-period)
    vol_series = pd.Series(volume)
    avg_vol = vol_series.rolling(window=20, min_periods=20).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 50  # for 12h EMA and BB calculation
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(ma_20[i]) or 
            np.isnan(std_20[i]) or np.isnan(avg_vol[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        # Trend filter: only trade in direction of 12h EMA
        trend_up = price > ema_50_12h_aligned[i]
        trend_down = price < ema_50_12h_aligned[i]
        
        if position == 0:
            # Long: price breaks above upper BB with volume filter and uptrend
            if price > upper_bb[i] and vol > 2.0 * avg_vol[i] and trend_up:
                position = 1
                signals[i] = position_size
            # Short: price breaks below lower BB with volume filter and downtrend
            elif price < lower_bb[i] and vol > 2.0 * avg_vol[i] and trend_down:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below middle band (mean reversion) or opposite BB
            if price < ma_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above middle band (mean reversion) or opposite BB
            if price > ma_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_Bollinger_Breakout_12hEMA_Volume"
timeframe = "4h"
leverage = 1.0